import jax
import jax.numpy as jnp

from envs.tfbind.prefix_tb_tfbind import prefix_tb_tfbind_trainer


def get_eval_fn(get_eval_rollout_fn, env, true_dist, cfg):
    """Evaluate proteins without attempting to enumerate the 20**60 states."""
    get_rollout = jax.jit(get_eval_rollout_fn)
    get_log_rewards = jax.jit(env.log_reward)
    logger = {
        "traj_length/max": [], "traj_length/mean": [],
        "traj_length/truncated_fraction": [],
    }

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
        if bool(jnp.any(completed)):
            log_rewards = get_log_rewards(terminal_states)
            logger["log_reward/mean"] = [jnp.mean(log_rewards[completed])]
            marginals = env.get_position_marginals(terminal_states[completed])
            logger["data/terminal_marginals"] = [marginals]
            logger.update(env.visualize(marginals, prefix="terminal_marginals"))
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
