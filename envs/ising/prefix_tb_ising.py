"""Prefix-TB training with uniform proposals and evaluation-only Ising references."""

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np

from envs.ising.metrics import correlation_error, correlation_profile, magnetization_error, sinkhorn_distance
from envs.ising.reference import get_reference_pair
from envs.tfbind.prefix_tb_tfbind import prefix_tb_tfbind_trainer


def get_path_rollout(key, model_state, params, states, env, rollout_max_length, is_forward=True):
    """Stream a stopped path and its log weight, without storing every lattice.

    log w = log R(x_T) - log L(x_0) + log P_B(path|x_T) - log P_F(path|x_0).
    Both conditional path probabilities include their respective stop action.
    Forward/backward expectations bound log(Z_R/Z_L). Censored paths return NaN.
    """
    if rollout_max_length < 1:
        raise ValueError("Evaluation rollout length must be positive")
    states = states.astype(jnp.int32)
    origin = states
    pf, pb, _ = model_state.apply_fn(params, origin)
    log_weights = (pb[:, -1] - env.log_initial_reward(origin) if is_forward
                   else env.log_reward(origin) - pf[:, -1])
    batch = jnp.arange(len(states))

    def cond(carry):
        _, done, _, _, _, step = carry
        return jnp.any(~done) & (step < rollout_max_length)

    def step(carry):
        current, done, key_gen, weights, lengths, iteration = carry
        log_pf, log_pb, _ = model_state.apply_fn(params, current)
        key_gen, action_key = jax.random.split(key_gen)
        action = jax.random.categorical(action_key, log_pf if is_forward else log_pb)
        next_states, next_done = jax.vmap(env.step)(current, action, done)
        reverse_action = jax.vmap(env.get_backward_action)(current, action)
        next_pf, next_pb, _ = model_state.apply_fn(params, next_states)
        if is_forward:
            edit_weight = next_pb[batch, reverse_action] - log_pf[batch, action]
            stop_weight = env.log_reward(current) - log_pf[:, -1]
        else:
            edit_weight = log_pb[batch, action] - next_pf[batch, reverse_action]
            stop_weight = log_pb[:, -1] - env.log_initial_reward(current)
        increment = jnp.where(action == env.stop_action, stop_weight, edit_weight)
        weights = weights + jnp.where(done, 0.0, increment)
        return next_states, next_done, key_gen, weights, lengths + (~done), iteration + 1

    initial = (states, jnp.zeros(len(states), dtype=bool), key, log_weights,
               jnp.zeros(len(states), dtype=jnp.int32), jnp.array(0))
    final, done, _, weights, lengths, _ = jax.lax.while_loop(cond, step, initial)
    return {
        'initial_states': origin if is_forward else final,
        'terminal_states': final if is_forward else origin,
        'log_weights': jnp.where(done, weights, jnp.nan),
        'lengths': jnp.where(done, lengths, rollout_max_length + 1),
        'completed': done,
    }


def get_eval_fn(get_eval_rollout_fn, env, true_dist, cfg):
    # The shared rollout starts uniformly and stores whole paths. Ising needs
    # source-law evaluation starts and streaming path ratios in both directions.
    del get_eval_rollout_fn, true_dist
    left_reference, right_reference = get_reference_pair(env, cfg.reference, cfg.seed)
    left_pool, right_pool = jnp.asarray(left_reference), jnp.asarray(right_reference)
    forward_rollout = jax.jit(partial(get_path_rollout, env=env,
                                    rollout_max_length=cfg.eval_rollout_max_length, is_forward=True))
    backward_rollout = jax.jit(partial(get_path_rollout, env=env,
                                     rollout_max_length=cfg.eval_rollout_max_length, is_forward=False))
    bound_offset = 0.0 if cfg.source_logZ is None else float(cfg.source_logZ)
    metric_names = ('elbo', 'eubo', 'sinkhorn', 'magnetization', 'correlation',
                    'sinkhorn_converged', 'sinkhorn_error', 'eval/num_completed',
                    'eval/num_backward_completed', 'eval/sinkhorn_num_samples',
                    'traj_length/max', 'traj_length/mean', 'traj_length/truncated_fraction',
                    'backward_traj_length/max', 'backward_traj_length/mean',
                    'backward_traj_length/truncated_fraction')
    logger = {name: [] for name in metric_names}
    logger['data/target_correlations'] = [correlation_profile(right_reference, env.lattice_size)]

    def short_eval(model_state, key):
        source_key, target_key, fwd_key, bwd_key, subset_key = jax.random.split(key, 5)
        starts = left_pool[jax.random.randint(source_key, (cfg.eval_batch_size,), 0, len(left_pool))]
        targets = right_pool[jax.random.randint(target_key, (cfg.eval_batch_size,), 0, len(right_pool))]
        forward = forward_rollout(fwd_key, model_state, model_state.params, starts)
        backward = backward_rollout(bwd_key, model_state, model_state.params, targets)
        completed = np.asarray(forward['completed'])
        bwd_completed = np.asarray(backward['completed'])
        terminal_states = np.asarray(forward['terminal_states'])
        samples = terminal_states[completed]
        values = {name: float('nan') for name in ('elbo', 'eubo', 'sinkhorn', 'magnetization', 'correlation', 'sinkhorn_error')}
        values.update({'sinkhorn_converged': False, 'eval/num_completed': int(completed.sum()),
                       'eval/num_backward_completed': int(bwd_completed.sum()),
                       'eval/sinkhorn_num_samples': 0})
        # Conditional averages after dropping unfinished paths are not evidence
        # bounds. Leave a gap in the metric until every path has terminated.
        if np.all(completed):
            values['elbo'] = float(np.asarray(forward['log_weights'], dtype=np.float64).mean()) + bound_offset
        if np.all(bwd_completed):
            values['eubo'] = float(np.asarray(backward['log_weights'], dtype=np.float64).mean()) + bound_offset
        if len(samples):
            values['magnetization'] = magnetization_error(samples, env.lattice_size)
            values['correlation'] = correlation_error(samples, right_reference, env.lattice_size)
            count = min(len(samples), cfg.sinkhorn.sample_size, len(targets))
            indices = np.asarray(jax.random.permutation(subset_key, len(samples)))[:count]
            values.update(sinkhorn_distance(samples[indices], np.asarray(targets)[:count],
                                           epsilon=cfg.sinkhorn.epsilon, threshold=cfg.sinkhorn.threshold,
                                           max_iterations=cfg.sinkhorn.max_iterations))
            values['eval/sinkhorn_num_samples'] = count
            logger.update(env.visualize(samples, prefix='terminal_states'))
            logger['data/model_correlations'] = [correlation_profile(samples, env.lattice_size)]
        else:
            logger['figures/terminal_states_vis'] = []
            logger['data/model_correlations'] = []
        for prefix, paths in (('traj_length', forward), ('backward_traj_length', backward)):
            lengths = np.asarray(paths['lengths'])
            values[f'{prefix}/max'] = int(lengths.max())
            values[f'{prefix}/mean'] = float(lengths.mean())
            values[f'{prefix}/truncated_fraction'] = float(1 - np.asarray(paths['completed']).mean())
        for name, value in values.items():
            logger[name].append(value)
        logger['data/terminal_states'] = [terminal_states]
        logger['data/initial_states'] = [np.asarray(starts)]
        logger['data/trajectory_lengths'] = [np.asarray(forward['lengths'])]
        logger['data/backward_trajectory_lengths'] = [np.asarray(backward['lengths'])]
        logger['data/forward_log_weights'] = [np.asarray(forward['log_weights'])]
        logger['data/backward_log_weights'] = [np.asarray(backward['log_weights'])]
        return logger

    return short_eval, logger


def prefix_tb_ising_trainer(cfg, experiment_logger):
    env = cfg.env
    if cfg.model.max_length != env.max_length or cfg.model.nchar != 2:
        raise ValueError("Ising model.max_length must equal env.lattice_size**2 and model.nchar must be 2")
    if min(cfg.batch_size, cfg.eval_batch_size, cfg.train_rollout_max_length,
           cfg.eval_rollout_max_length, cfg.sinkhorn.sample_size) < 1:
        raise ValueError("Batch sizes, rollout lengths and Sinkhorn sample size must be positive")
    if cfg.source_logZ is not None and not math.isfinite(cfg.source_logZ):
        raise ValueError("source_logZ must be finite or null")
    if cfg.init_logZ is None:
        # Ground-state approximation for the ratio; neither source samples nor
        # a uniform-proposal estimate of an exponentially concentrated Z is used.
        cfg.init_logZ = 2 * env.max_length * env.coupling * (env.beta_right - env.beta_left)
    if cfg.flow_penalty_log_scale is None:
        cfg.flow_penalty_log_scale = -2 * env.max_length * env.coupling * env.beta_right
    experiment_logger.log_metrics({'logZ_ratio_initial': cfg.init_logZ,
                                   'flow_penalty_log_scale': cfg.flow_penalty_log_scale}, step=0)
    return prefix_tb_tfbind_trainer(cfg, experiment_logger, eval_fn_factory=get_eval_fn)
