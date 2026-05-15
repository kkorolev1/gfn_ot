import jax
import jax.numpy as jnp
import jax.nn as nn

import optax

import flax.linen as flax_nn
import equinox
from flax.training.train_state import TrainState
from flax.traverse_util import path_aware_map
from functools import partial
import matplotlib.pyplot as plt

from envs.hypergrid.hypergrid import Hypergrid
from envs.hypergrid.buffer import build_terminal_state_buffer
from utils.helper import extract_last_entry, log1mexp


class Model(flax_nn.Module):
    dim: int = 2
    side: int = 10
    num_layers: int = 2
    num_hid: int = 64

    weight_init: float = 1e-8
    bias_init: float = 0.0

    def setup(self):
        self.num_actions = 4 * self.dim + 2

        self.model = flax_nn.Sequential(
            [
                flax_nn.Sequential([flax_nn.Dense(self.num_hid), flax_nn.gelu])
                for _ in range(self.num_layers)
            ]
            + [
                flax_nn.Dense(
                    self.num_actions,
                    kernel_init=flax_nn.initializers.constant(1e-8),
                    bias_init=flax_nn.initializers.zeros_init(),
                )
            ]
        )

    def __call__(self, states, log_rewards=None):
        logits = self.model(states)
        mask_minus = states == 0
        mask_plus = states == self.side - 1

        forward_logits, backward_logits = jnp.split(logits, [2 * self.dim + 1], axis=-1)
        mask = jnp.concatenate(
            (mask_minus, mask_plus, jnp.zeros((*logits.shape[:-1], 1), dtype=bool)),
            axis=-1,
        )
        forward_logits = jnp.where(mask, -float("inf"), forward_logits)
        log_pfs = nn.log_softmax(forward_logits, axis=-1)

        mask = jnp.concatenate(
            (mask_plus, mask_minus, jnp.zeros((*logits.shape[:-1], 1), dtype=bool)),
            axis=-1,
        )
        backward_logits = jnp.where(mask, -float("inf"), backward_logits)
        log_pbs = nn.log_softmax(backward_logits, axis=-1)

        if log_rewards is None:
            log_flows = jnp.zeros_like(log_pfs[..., 0])
        else:
            log_flows = log_rewards - log_pfs[..., -1]

        return log_pfs, log_pbs, log_flows


def model_label_map(path, _):
    if "logZ" in path:
        return "logZ_optim"
    else:
        return "model_optim"


def init_model(key_gen, cfg):
    def build_lr_schedule(base_lr):
        sched_cfg = getattr(cfg, "lr_schedule", None)
        if sched_cfg is None:
            return lambda step: base_lr

        sched_type = getattr(sched_cfg, "type", "constant")
        match sched_type:
            case "constant":
                return lambda step: base_lr
            case "multistep":
                milestones_arr = (
                    jnp.array(sched_cfg.milestones, dtype=jnp.int32)
                    if len(sched_cfg.milestones) > 0
                    else None
                )

                def multistep_fn(step):
                    if milestones_arr is None:
                        num_decays = 0
                    else:
                        num_decays = jnp.sum(step >= milestones_arr)
                    return base_lr * (sched_cfg.gamma**num_decays)

                return multistep_fn
            case "cosine":
                decay_steps = max(cfg.train_num_steps, 1)
                return optax.cosine_decay_schedule(
                    init_value=base_lr,
                    decay_steps=decay_steps,
                    alpha=sched_cfg.end_factor,
                )
            case _:
                raise ValueError(f"Invalid learning rate scheduler type: {sched_type}")

    model = Model(**cfg.model)
    key, key_gen = jax.random.split(key_gen)
    params = model.init(
        key, jnp.ones([cfg.batch_size, cfg.env.dim]), jnp.ones([cfg.batch_size])
    )
    params["params"] = {**params["params"], "logZ": jnp.array((cfg.init_logZ,))}
    optimizers_map = {
        "model_optim": optax.adam(learning_rate=build_lr_schedule(cfg.lr)),
        "logZ_optim": optax.adam(learning_rate=build_lr_schedule(cfg.logZ_lr)),
    }
    param_labels = path_aware_map(model_label_map, params)
    partitioned_optimizer = optax.multi_transform(optimizers_map, param_labels)
    optimizer = optax.chain(
        optax.zero_nans(),
        (
            optax.clip_by_global_norm(cfg.grad_clip)
            if cfg.grad_clip > 0
            else optax.identity()
        ),
        partitioned_optimizer,
    )
    model_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return model_state


def per_sample_get_train_rollout(
    key,
    model_state,
    params,
    state: jnp.ndarray,
    env: Hypergrid,
    rollout_max_length: int,
    is_forward_rollout: bool = True,
):
    def simulate_forward_rollout(data, per_step_input):
        state, key_gen = data
        state = jax.lax.stop_gradient(state)
        log_reward = env.log_reward(state)
        log_pf_dist, _, log_flow = model_state.apply_fn(params, state, log_reward)
        key, key_gen = jax.random.split(key_gen)
        action = jax.random.categorical(key, log_pf_dist[..., :-1])
        log_pf = log_pf_dist[action]
        state_next, _ = env.step(state, action, jnp.array(False))
        state_next = jax.lax.stop_gradient(state_next)
        _, log_pb_dist, _ = model_state.apply_fn(params, state_next)
        log_pb = log_pb_dist[action]
        data_next = (state_next, key_gen)
        per_step_output = (log_pf, log_pb, log_flow, log_reward)
        return data_next, per_step_output

    def simulate_backward_rollout(data, per_step_input):
        state_next, key_gen = data
        state_next = jax.lax.stop_gradient(state_next)
        _, log_pb_dist, _ = model_state.apply_fn(params, state_next)
        key, key_gen = jax.random.split(key_gen)
        action = jax.random.categorical(key, log_pb_dist[..., :-1])
        log_pb = log_pb_dist[action]
        state, _ = env.step_backward(state_next, action, jnp.array(False))
        log_reward = env.log_reward(state)
        state = jax.lax.stop_gradient(state)
        log_pf_dist, _, log_flow = model_state.apply_fn(params, state, log_reward)
        log_pf = log_pf_dist[action]
        data_next = (state, key_gen)
        per_step_output = (log_pf, log_pb, log_flow, log_reward)
        return data_next, per_step_output

    if is_forward_rollout:
        init_state = state
        aux = (init_state, key)
        aux, per_step_output = jax.lax.scan(
            simulate_forward_rollout, aux, jnp.arange(rollout_max_length)
        )
        terminal_state, _ = aux
    else:
        terminal_state = state
        aux = (terminal_state, key)
        aux, per_step_output = jax.lax.scan(
            simulate_backward_rollout, aux, jnp.arange(rollout_max_length)[::-1]
        )
        init_state, _ = aux

    log_pf, log_pb, log_flow, log_reward = per_step_output
    return terminal_state, init_state, log_pf, log_pb, log_flow, log_reward


def get_train_rollout(
    key_gen,
    model_state,
    params,
    env: Hypergrid,
    batch_size: int,
    rollout_max_length: int,
    initial_dist,
    is_forward_rollout: bool = True,
    terminal_states=None,
    terminal_log_rewards=None,
):
    initial_dist_sample_fn, initial_dist_log_prob_fn = initial_dist
    if is_forward_rollout:
        key, key_gen = jax.random.split(key_gen)
        states = initial_dist_sample_fn(key, (batch_size,))
    else:
        states = terminal_states

    keys = jax.random.split(key_gen, num=batch_size)
    (
        terminal_states,
        init_states,
        log_pfs,
        log_pbs,
        log_flows,
        log_rewards,
    ) = jax.vmap(
        per_sample_get_train_rollout,
        in_axes=(0, None, None, 0, None, None, None),
    )(
        keys,
        model_state,
        params,
        states,
        env,
        rollout_max_length,
        is_forward_rollout,
    )
    if not is_forward_rollout:
        log_pfs = log_pfs[:, ::-1]
        log_pbs = log_pbs[:, ::-1]
        log_flows = log_flows[:, ::-1]
        log_rewards = log_rewards[:, ::-1]
    # We need to account transition (s0, s1) and calculate log probs for it
    log_pfs = jnp.concatenate(
        [initial_dist_log_prob_fn(init_states)[:, None], log_pfs], axis=1
    )
    _, log_pb_initial, _ = model_state.apply_fn(
        params, jax.lax.stop_gradient(init_states), log_rewards[:, 0]
    )
    log_pbs = jnp.concatenate([log_pb_initial[..., -1][:, None], log_pbs], axis=1)
    # We need to calculate log flow for the last non-terminal state
    if terminal_log_rewards is None:
        terminal_log_rewards = env.log_reward(terminal_states)
    terminal_states = jax.lax.stop_gradient(terminal_states)
    _, _, terminal_log_flows = model_state.apply_fn(
        params, terminal_states, terminal_log_rewards
    )
    log_flows = jnp.concatenate([log_flows, terminal_log_flows[:, None]], axis=1)
    log_rewards = jnp.concatenate([log_rewards, terminal_log_rewards[:, None]], axis=1)
    log_pfs_over_pbs = log_pfs - log_pbs

    return (terminal_states, log_pfs_over_pbs, log_rewards, log_flows)


def prefix_tb_loss_fn(
    key,
    model_state,
    params,
    get_train_rollout_fn,
    reg_coef: float = 0.0,
    use_weights: bool = False,
):
    terminal_states, log_pfs_over_pbs, log_rewards, log_flows = get_train_rollout_fn(
        key, model_state, params
    )
    batch_size, rollout_length = log_pfs_over_pbs.shape
    logZ = params["params"]["logZ"]
    discrepancy = logZ + jnp.cumsum(log_pfs_over_pbs, axis=-1) - log_flows

    if use_weights:
        log_probs_stop = log_rewards - log_flows
        log_probs_nonstop = log1mexp(-log_probs_stop)
        log_weights = (
            jnp.cumsum(
                jnp.concatenate(
                    [
                        jnp.zeros((batch_size, 1)),
                        log_probs_nonstop[:, :-1],
                    ],
                    axis=1,
                ),
                axis=1,
            )
            + log_probs_stop
        )
        log_weights = jax.lax.stop_gradient(log_weights)
        weights = nn.softmax(log_weights)
    else:
        weights = jnp.ones((batch_size, 1)) / rollout_length

    tb_losses = jnp.square(discrepancy) * weights
    flow_penalties = reg_coef * jnp.exp(log_flows) * weights
    losses = tb_losses.sum(-1) + flow_penalties.sum(-1)
    return jnp.mean(losses), (
        terminal_states,
        log_rewards[:, -1],
        jax.lax.stop_gradient(losses),
    )


def per_sample_get_eval_rollout(
    key,
    model_state,
    params,
    state: jnp.ndarray,
    env: Hypergrid,
    rollout_max_length: int,
):
    def cond_fun(carry):
        data, _ = carry
        _, is_terminal, _, step = data
        return (jnp.any(~is_terminal)) & (step < rollout_max_length)

    def simulate_forward_rollout(carry, force_stop=False):
        data, data_hist = carry
        state, is_terminal, key_gen, step = data
        trajectories, terminals_mask = data_hist
        state = jax.lax.stop_gradient(state)
        log_pf_dist, _, _ = model_state.apply_fn(params, state)
        key, key_gen = jax.random.split(key_gen)
        action = jax.random.categorical(key, log_pf_dist)
        if force_stop:
            action = env.stop_action
        state_next, is_terminal_next = env.step(state, action, is_terminal)
        trajectories = trajectories.at[step].set(state)
        terminals_mask = terminals_mask.at[step].set(is_terminal)
        data_next = (state_next, is_terminal_next, key_gen, step + 1)
        data_hist = (trajectories, terminals_mask)
        return (data_next, data_hist)

    d = state.shape[-1]
    trajectories = jnp.zeros((rollout_max_length + 1, d))
    terminals_mask = jnp.ones((rollout_max_length + 1,), dtype=bool)
    data_hist = (trajectories, terminals_mask)
    data_init = (state, jnp.array(False), key, 0)
    carry = (data_init, data_hist)

    carry = equinox.internal.while_loop(
        cond_fun,
        simulate_forward_rollout,
        carry,
        max_steps=rollout_max_length,
        kind="checkpointed",
    )
    carry = simulate_forward_rollout(carry, force_stop=True)
    _, state_hist = carry
    return state_hist


def get_eval_rollout(
    key_gen,
    model_state,
    params,
    env: Hypergrid,
    batch_size: int,
    rollout_max_length: int,
    initial_dist,
):
    initial_dist_sample_fn, _ = initial_dist
    key, key_gen = jax.random.split(key_gen)
    states = initial_dist_sample_fn(key, (batch_size,))

    keys = jax.random.split(key_gen, num=batch_size)
    (trajectories, terminals_mask) = jax.vmap(
        per_sample_get_eval_rollout,
        in_axes=(0, None, None, 0, None, None),
    )(
        keys,
        model_state,
        params,
        states,
        env,
        rollout_max_length,
    )
    trajectories_length = (~terminals_mask).sum(axis=1)
    return trajectories, trajectories_length


def compute_empirical_dist(samples, dim, side):
    def add_sample(rewards, s):
        idx = tuple(s.astype(jnp.int32))
        return rewards.at[idx].add(1.0)

    rewards = jax.lax.fori_loop(
        0,
        samples.shape[0],
        lambda i, rewards: add_sample(rewards, samples[i]),
        jnp.zeros((side,) * dim, dtype=jnp.float32),
    )
    return rewards / jnp.sum(rewards)


def get_eval_fn(get_eval_rollout_fn, env, true_dist, cfg):
    get_eval_forward_rollout = jax.jit(get_eval_rollout_fn)

    logger = {"tv": [], "traj_length/max": [], "traj_length/mean": []}
    all_samples = None

    def short_eval(model_state, key):
        if isinstance(model_state, tuple):
            model_state1, model_state2 = model_state
            params = (model_state1.params, model_state2.params)
        else:
            params = (model_state.params,)
        trajectories, trajectories_length = get_eval_forward_rollout(
            key, model_state, *params
        )
        terminal_states = trajectories[
            jnp.arange(trajectories.shape[0]), trajectories_length - 1
        ]
        nonlocal all_samples
        if all_samples is None:
            all_samples = terminal_states
        else:
            all_samples = jnp.concatenate([all_samples, terminal_states], axis=0)

        empirical_dist = compute_empirical_dist(all_samples[-10000:], env.dim, env.side)
        tv = jnp.abs(true_dist - empirical_dist).sum()
        logger["tv"].append(tv)
        logger["traj_length/max"].append(jnp.max(trajectories_length))
        logger["traj_length/mean"].append(jnp.mean(trajectories_length))
        logger.update(
            env.visualize(
                empirical_dist,
                prefix="empirical_dist",
            )
        )
        return logger

    return short_eval, logger


def prefix_tb_hypergrid_trainer(cfg, comet_exp=None):
    key_gen = jax.random.PRNGKey(cfg.seed)

    env: Hypergrid = cfg.env
    batch_size = cfg.batch_size
    train_rollout_max_length = cfg.train_rollout_max_length
    eval_rollout_max_length = cfg.eval_rollout_max_length

    key, key_gen = jax.random.split(key_gen)
    model_state = init_model(key, cfg)

    true_rewards = env.get_grid_rewards()
    true_logZ = jnp.sum(true_rewards)
    true_dist = true_rewards / true_logZ
    print(f"True logZ: {true_logZ:.4f}")

    def get_initial_dist(r_inner=0.32, r_outer=0.45, offset=0.25):
        # Uniform initial distribution
        # def sample(key, sample_shape=()):
        #     return jax.random.randint(
        #         key,
        #         shape=(*sample_shape, env.dim),
        #         minval=0,
        #         maxval=env.side,
        #     )

        # def log_prob(states):
        #     return -env.dim * jnp.log(env.side * jnp.ones(states.shape[:-1]))

        # Moon initial distribution
        side, dim = env.side, env.dim

        def log_prob(states):
            z = states.astype(jnp.float64) / (side - 1)

            c = jnp.full((dim,), 0.5, dtype=jnp.float64)
            e1 = jnp.zeros((dim,), dtype=jnp.float64).at[0].set(1.0)

            in_outer = jnp.sum((z - c) ** 2, axis=1) <= r_outer**2
            in_inner = jnp.sum((z - (c - offset * e1)) ** 2, axis=1) <= r_inner**2
            moon = jnp.logical_and(in_outer, jnp.logical_not(in_inner))

            back = c + (r_outer / 2.0) * e1
            arc_dist = jnp.linalg.norm(z - back, axis=1)

            centeredness = jnp.clip(1.0 - arc_dist / r_outer, 0.0, 1.0)
            prob = moon.astype(jnp.float64) * (0.5 + 2.0 * centeredness) + 1e-3
            return jnp.log(prob)

        ranges = [jnp.arange(side, dtype=jnp.int32) for _ in range(dim)]
        mesh = jnp.meshgrid(*ranges, indexing="ij")
        all_states = jnp.stack(mesh, axis=-1).reshape(-1, dim)

        logprobs = log_prob(all_states)
        weights = jnp.exp(logprobs)
        probs = weights / jnp.sum(weights)

        def sample(key, sample_shape=()):
            flat_indices = jax.random.choice(
                key, all_states.shape[0], shape=sample_shape, replace=True, p=probs
            )
            sampled_states = all_states[flat_indices]
            return sampled_states

        return sample, log_prob

    initial_dist = get_initial_dist()

    get_train_rollout_base = partial(
        get_train_rollout,
        env=env,
        batch_size=batch_size,
        rollout_max_length=train_rollout_max_length,
        initial_dist=initial_dist,
    )

    get_eval_rollout_base = partial(
        get_eval_rollout,
        env=env,
        batch_size=batch_size,
        rollout_max_length=eval_rollout_max_length,
        initial_dist=initial_dist,
    )

    buffer_cfg = cfg.buffer
    use_buffer = buffer_cfg.use
    buffer = buffer_state = None
    if use_buffer:
        buffer = build_terminal_state_buffer(
            dim=env.dim,
            max_length=buffer_cfg.max_length_in_batches * batch_size,
            prioritize_by=buffer_cfg.prioritize_by,
            sampling_method=buffer_cfg.sampling_method,
            rank_k=buffer_cfg.rank_k,
        )
        buffer_state = buffer.init(
            dtype=jnp.float32, device=jax.devices(jax.default_backend())[0]
        )

    loss_fn_base = partial(
        prefix_tb_loss_fn, reg_coef=cfg.reg_coef, use_weights=cfg.use_weights
    )

    @partial(jax.jit)
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_fwd_grad_fn(key, model_state, params):
        get_train_forward_rollout = partial(
            get_train_rollout_base, is_forward_rollout=True
        )
        return loss_fn_base(key, model_state, params, get_train_forward_rollout)

    @partial(jax.jit)
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_bwd_grad_fn(
        key, model_state, params, terminal_states, terminal_log_rewards
    ):
        get_train_backward_rollout = partial(
            get_train_rollout_base,
            is_forward_rollout=False,
            terminal_states=terminal_states,
            terminal_log_rewards=terminal_log_rewards,
        )
        return loss_fn_base(key, model_state, params, get_train_backward_rollout)

    eval_fn, logger = get_eval_fn(partial(get_eval_rollout_base), env, true_dist, cfg)

    for step in range(cfg.train_num_steps):
        if not use_buffer or step % (buffer_cfg.replay_ratio + 1) == 0:
            key, key_gen = jax.random.split(key_gen)
            grads, (terminal_states, terminal_log_rewards, losses) = loss_fwd_grad_fn(
                key, model_state, model_state.params
            )
            model_state = model_state.apply_gradients(grads=grads)

            # Add samples to buffer
            if use_buffer:
                buffer_state = buffer.add(
                    buffer_state,
                    terminal_states,
                    terminal_log_rewards,
                    losses,
                )
        else:
            key, key_gen = jax.random.split(key_gen)
            terminal_states, terminal_log_rewards, indices = buffer.sample(
                buffer_state, key, batch_size
            )

            key, key_gen = jax.random.split(key_gen)
            grads, _ = loss_bwd_grad_fn(
                key,
                model_state,
                model_state.params,
                terminal_states,
                terminal_log_rewards,
            )
            model_state = model_state.apply_gradients(grads=grads)

        if cfg.use_comet:
            comet_exp.log_metrics(
                {
                    "loss": jnp.mean(losses),
                    "logZ_learned": model_state.params["params"]["logZ"],
                },
                step=step,
            )

        if (step % cfg.eval_frequency == 0) or (step == cfg.train_num_steps - 1):
            key, key_gen = jax.random.split(key_gen)
            logger.update(eval_fn(model_state, key))

            print(
                f"[{step}/{cfg.train_num_steps}] "
                f"Loss: {jnp.mean(losses):.4f}, "
                f"TV: {logger['tv'][-1]:.4f}, "
                f"Max Len: {logger['traj_length/max'][-1]:.4f}, "
                f"Mean Len: {logger['traj_length/mean'][-1]:.4f}"
            )

            if use_buffer:
                key, key_gen = jax.random.split(key_gen)
                buffer_terminal_states, _, _ = buffer.sample(
                    buffer_state,
                    key,
                    batch_size,
                )
                buffer_empirical_dist = compute_empirical_dist(
                    buffer_terminal_states, env.dim, env.side
                )
                logger.update(
                    env.visualize(buffer_empirical_dist, prefix="buffer_empirical_dist")
                )

            if cfg.use_comet:
                last_entry = extract_last_entry(logger)
                metrics = {}
                for key, value in last_entry.items():
                    if isinstance(value, plt.Figure):
                        comet_exp.log_figure(figure=value, figure_name=key, step=step)
                    else:
                        metrics[key] = value
                comet_exp.log_metrics(metrics, step=step)
