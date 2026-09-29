from functools import partial
import math

import jax
import jax.numpy as jnp

from envs.amp.autoregressive import AutoregressiveSampler
from envs.tfbind.prefix_tb_tfbind import prefix_tb_tfbind_trainer
from libs.gfnx.baselines.utils.amp_reward import resolve_sampler_checkpoint


@partial(jax.jit, static_argnames=("epsilon", "threshold", "max_iterations"))
def _sinkhorn_divergence_hamming(x, y, epsilon, threshold, max_iterations):
    from ott.geometry.geometry import Geometry
    from ott.tools.sinkhorn_divergence import sinkhorn_divergence

    def cost(lhs, rhs):
        return jnp.sum(lhs[:, None, :] != rhs[None, :, :], axis=-1, dtype=jnp.float32)

    result = sinkhorn_divergence(
        Geometry,
        cost_matrix=(cost(x, y), cost(x, x), cost(y, y)),
        epsilon=epsilon, relative_epsilon=False, scale_cost=1.0,
        sinkhorn_kwargs={
            "threshold": threshold, "max_iterations": max_iterations, "lse_mode": True,
        },
    )
    return result.divergence, jnp.all(jnp.asarray(result.converged))


def sinkhorn_divergence_hamming(x, y, epsilon=1.0, threshold=1e-4, max_iterations=2000):
    """Debiased entropic OT between empirical laws, with unit-cost residue edits.

    S_eps(x,y) = OT_eps(x,y) - (OT_eps(x,x) + OT_eps(y,y))/2.
    All three terms include entropy and use the same absolute epsilon.
    """
    x, y = jnp.asarray(x), jnp.asarray(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Sinkhorn inputs must be batches of equal-length sequences")
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Sinkhorn inputs must contain at least one sample")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Sinkhorn epsilon must be finite and positive")
    if not math.isfinite(threshold) or threshold <= 0 or max_iterations < 1:
        raise ValueError("Sinkhorn threshold and max_iterations must be positive")
    value, converged = _sinkhorn_divergence_hamming(x, y, epsilon, threshold, max_iterations)
    value = float(value)
    if not bool(converged) or not math.isfinite(value):
        raise RuntimeError("Sinkhorn did not converge; increase sinkhorn.max_iterations or epsilon")
    if value < -1e-4:
        raise RuntimeError(f"Numerically negative Sinkhorn divergence: {value}; tighten threshold")
    return max(value, 0.0)  # Remove only floating-point roundoff near zero.


def get_eval_fn(get_eval_rollout_fn, env, true_dist, cfg):
    """Evaluate proteins without attempting to enumerate the 20**60 states."""
    get_rollout = jax.jit(get_eval_rollout_fn)
    get_log_rewards = jax.jit(env.log_reward)
    logger = {
        "traj_length/max": [], "traj_length/mean": [],
        "traj_length/truncated_fraction": [],
    }
    sd_cfg = getattr(cfg, "sinkhorn", None)
    use_sd = sd_cfg is not None and sd_cfg.target_checkpoint is not None
    if use_sd:
        target_checkpoint = resolve_sampler_checkpoint(sd_cfg.target_checkpoint)
        target_sampler = AutoregressiveSampler.load(target_checkpoint)
        if target_sampler.max_length != env.max_length or target_sampler.nchar != env.nchar:
            raise ValueError("The Sinkhorn target checkpoint must match the AMP length/alphabet")
        max_samples = min(int(sd_cfg.sample_size), int(cfg.eval_batch_size))
        if max_samples < 1:
            raise ValueError("Sinkhorn sample_size and eval_batch_size must be positive")
        get_target_samples = jax.jit(lambda key: target_sampler.sample(key, (max_samples,)))
        # A separate random stream from training, with independent target draws.
        key_a, key_b = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(cfg.seed), 1729))
        target_a, target_b = get_target_samples(key_a), get_target_samples(key_b)
        sample_source, _ = env.get_initial_dist()
        get_source_samples = jax.jit(lambda key: sample_source(key, (max_samples,)))
        # Independent of rollout starts; reuse the same left sample set in both transports.
        source = get_source_samples(jax.random.fold_in(jax.random.PRNGKey(cfg.seed), 1731))
        divergence = partial(
            sinkhorn_divergence_hamming, epsilon=float(sd_cfg.epsilon),
            threshold=float(sd_cfg.threshold), max_iterations=int(sd_cfg.max_iterations),
        )
        target_baselines = {}

        def target_baseline(count):
            if count not in target_baselines:
                target_baselines[count] = (
                    divergence(target_a[:count], target_b[:count]),
                    divergence(source[:count], target_a[:count]),
                )
            return target_baselines[count]

        target_sd, target_transport_sd = target_baseline(max_samples)
        logger.update({
            "sd": [], "sd_num_samples": [],
            "target_sd": [target_sd],
            "target_sd_num_samples": [max_samples],
            "transport_sd": [], "target_transport_sd": [target_transport_sd],
        })

    def short_eval(model_state, key):
        trajectories, lengths = get_rollout(key, model_state, model_state.params)
        terminal_states = trajectories[jnp.arange(trajectories.shape[0]), lengths - 1]
        completed = lengths <= cfg.eval_rollout_max_length
        logger["traj_length/max"].append(jnp.max(lengths))
        logger["traj_length/mean"].append(jnp.mean(lengths))
        logger["traj_length/truncated_fraction"].append(jnp.mean(~completed))
        logger["data/terminal_states"] = [terminal_states]
        logger["data/trajectory_lengths"] = [lengths]
        logger["log_reward/mean"] = []
        logger["data/terminal_marginals"] = []
        logger["figures/terminal_marginals_vis"] = []
        if use_sd:
            # Never re-log a previous model's SD when this batch is all forced stops.
            logger["sd"] = []
            logger["transport_sd"] = []
            logger["sd_num_samples"] = [0]
        if bool(jnp.any(completed)):
            log_rewards = get_log_rewards(terminal_states)
            logger["log_reward/mean"] = [jnp.mean(log_rewards[completed])]
            marginals = env.get_position_marginals(terminal_states[completed])
            logger["data/terminal_marginals"] = [marginals]
            logger.update(env.visualize(marginals, prefix="terminal_marginals"))
            if use_sd:
                samples = terminal_states[completed]
                count = min(samples.shape[0], max_samples)
                if samples.shape[0] > count:
                    sample_key = jax.random.fold_in(key, 1730)
                    indices = jax.random.choice(sample_key, samples.shape[0], (count,), replace=False)
                    samples = samples[indices]
                logger["sd"] = [divergence(samples, target_a[:count])]
                logger["transport_sd"] = [divergence(source[:count], samples)]
                # Match all comparisons to the accepted model sample count.
                target_sd, target_transport_sd = target_baseline(count)
                logger["target_sd"] = [target_sd]
                logger["target_transport_sd"] = [target_transport_sd]
                logger["sd_num_samples"] = [count]
                logger["target_sd_num_samples"] = [count]
        return logger

    return short_eval, logger


def prefix_tb_amp_trainer(cfg, experiment_logger):
    # Keep rewards as p(x)**beta. Only rescale the flow penalty, whose raw
    # exponentials otherwise underflow for length-60 sequences in float32.
    if cfg.init_logZ is None or cfg.flow_penalty_log_scale is None:
        sample, log_prob = cfg.env.get_initial_dist()

        @jax.jit
        def calibrate(key):
            states = sample(key, (cfg.batch_size,))
            log_p = log_prob(states)
            log_r = cfg.env.beta * log_p
            log_z = jax.nn.logsumexp(log_r - log_p) - jnp.log(cfg.batch_size)
            return log_z, -jnp.max(log_r)

        log_z, log_scale = calibrate(jax.random.PRNGKey(cfg.seed))
        if cfg.init_logZ is None:
            # Importance estimate E_p[p(x)**(beta-1)], used only to initialize Z.
            cfg.init_logZ = float(log_z)
        if cfg.flow_penalty_log_scale is None:
            cfg.flow_penalty_log_scale = float(log_scale)
    experiment_logger.log_metrics({
        "logZ_initial": cfg.init_logZ,
        "flow_penalty_log_scale": cfg.flow_penalty_log_scale,
    }, step=0)
    return prefix_tb_tfbind_trainer(cfg, experiment_logger, eval_fn_factory=get_eval_fn)
