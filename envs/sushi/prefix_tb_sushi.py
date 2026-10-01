"""Use the shared prefix-TB trainer with normalized ranking endpoints."""

import jax
import jax.numpy as jnp
import numpy as np

from envs.sushi.sushi import kendall_distance, plackett_luce_kendall_ot
from envs.tfbind.prefix_tb_tfbind import prefix_tb_tfbind_trainer
from utils.helper import build_evaluation_buffer


def get_eval_fn(get_eval_rollout_fn, env, true_dist, cfg):
    rollout = jax.jit(get_eval_rollout_fn)
    update_buffer, baseline = build_evaluation_buffer(true_dist, cfg.eval_buffer_size, cfg.seed)
    logger = {'tv': [], 'target/tv': [baseline['tv']],
              'traj_length/max': [], 'traj_length/mean': [],
              'traj_length/truncated_fraction': []}
    optimal_cost = plackett_luce_kendall_ot(env.left, env.right)
    logger['target/ot_cost'] = [optimal_cost]
    logger['target/mean_traj_length'] = [optimal_cost + 1]
    print(f'Exact Kendall OT: {optimal_cost:.6f}; optimal mean length including stop: {optimal_cost + 1:.6f}')
    logger.update(env.visualize(true_dist, prefix='target_dist'))

    def evaluate(model_state, key):
        trajectories, lengths = rollout(key, model_state, model_state.params)
        terminal_states = trajectories[jnp.arange(len(lengths)), lengths - 1]
        completed = lengths <= cfg.eval_rollout_max_length
        samples = terminal_states[completed]
        # The generic TV buffer uses coordinates into a tensor; a permutation's
        # Lehmer index is its coordinate into our flat vector of n! probabilities.
        empirical, metrics = update_buffer(env.get_state_indices(samples)[:, None])
        for name, value in metrics.items():
            logger[name].append(value)
        logger['traj_length/max'].append(jnp.max(lengths))
        logger['traj_length/mean'].append(jnp.mean(lengths))
        logger['traj_length/truncated_fraction'].append(jnp.mean(~completed))
        logger['eval/num_completed'] = [jnp.sum(completed)]
        logger['transport/mean_swaps'] = []
        logger['transport/kendall'] = []
        if len(samples):
            # Length includes stop; OT's unit cost counts only actual swaps.
            logger['transport/mean_swaps'] = [jnp.mean(lengths[completed] - 1)]
            logger['transport/kendall'] = [jnp.mean(kendall_distance(trajectories[completed, 0], samples))]
        logger['data/initial_states'] = [trajectories[:, 0]]
        logger['data/terminal_states'] = [terminal_states]
        logger['data/trajectory_lengths'] = [lengths]
        if empirical is not None:
            logger['data/terminal_dist'] = [empirical]
            logger.update(env.visualize(empirical, prefix='terminal_dist'))
        return logger

    return evaluate, logger


def prefix_tb_sushi_trainer(cfg, experiment_logger):
    env = cfg.env
    if cfg.model.max_length != env.max_length or cfg.model.nchar != env.nchar or not cfg.model.adjacent_swaps:
        raise ValueError('SUSHI policy must use adjacent_swaps and match the checkpoint ranking length')
    if cfg.init_logZ != 0 or cfg.logZ_lr != 0:
        raise ValueError('Normalized SUSHI rewards require fixed logZ: init_logZ=0 and logZ_lr=0')
    if min(cfg.batch_size, cfg.eval_batch_size, cfg.train_rollout_max_length,
           cfg.eval_rollout_max_length, cfg.eval_buffer_size) < 1:
        raise ValueError('Batch sizes, rollout lengths and evaluation capacity must be positive')
    experiment_logger.log_array('left_worth', np.exp(env.left.log_worth), step=0)
    experiment_logger.log_array('right_worth', np.exp(env.right.log_worth), step=0)
    return prefix_tb_tfbind_trainer(cfg, experiment_logger, eval_fn_factory=get_eval_fn)
