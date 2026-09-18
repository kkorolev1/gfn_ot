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

from envs.tfbind.tfbind import TFBind8Environment
from envs.tfbind.buffer import build_terminal_state_buffer
from envs.tfbind.evaluation import build_exact_evaluator, build_sample_evaluator
from utils.helper import extract_last_entry, log1mexp


class Model(flax_nn.Module):
    max_length: int = 8
    nchar: int = 4
    num_layers: int = 2
    num_hid: int = 64

    weight_init: float = 1e-8
    bias_init: float = 0.0

    def setup(self):
        self.num_actions = 2 * (self.max_length * self.nchar + 1)

        self.model = flax_nn.Sequential(
            [
                flax_nn.Sequential([flax_nn.Dense(self.num_hid), flax_nn.gelu])
                for _ in range(self.num_layers)
            ]
            + [
                flax_nn.Dense(
                    self.num_actions,
                    kernel_init=flax_nn.initializers.constant(self.weight_init),
                    bias_init=flax_nn.initializers.constant(self.bias_init),
                )
            ]
        )

    def __call__(self, states, log_rewards=None):
        encoded = nn.one_hot(states.astype(jnp.int32), self.nchar)
        encoded = encoded.reshape((*states.shape[:-1], self.max_length * self.nchar))
        logits = self.model(encoded)
        forward_logits, backward_logits = jnp.split(logits, 2, axis=-1)
        mask = encoded.astype(bool)
        mask = jnp.concatenate(
            (mask, jnp.zeros((*states.shape[:-1], 1), dtype=bool)), axis=-1
        )
        log_pfs = nn.log_softmax(jnp.where(mask, -jnp.inf, forward_logits), axis=-1)
        log_pbs = nn.log_softmax(jnp.where(mask, -jnp.inf, backward_logits), axis=-1)

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
        key, jnp.ones([cfg.batch_size, cfg.env.max_length]), jnp.ones([cfg.batch_size])
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
    env: TFBind8Environment,
    rollout_max_length: int,
    is_forward_rollout: bool = True,
):
    state = state.astype(jnp.int32)

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
        backward_action = env.get_backward_action(state, action)
        log_pb = log_pb_dist[backward_action]
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
        forward_action = env.get_backward_action(state_next, action)
        log_pf = log_pf_dist[forward_action]
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
    env: TFBind8Environment,
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
    env: TFBind8Environment,
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
    trajectories = jnp.zeros((rollout_max_length + 1, d), dtype=jnp.int32)
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
    env: TFBind8Environment,
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


def get_eval_fn(get_eval_rollout_fn, env, true_dist, cfg, true_log_rewards=None):
    get_eval_forward_rollout = jax.jit(get_eval_rollout_fn)
    sample_eval, target_metrics = build_sample_evaluator(
        true_dist, cfg.batch_size, cfg.seed
    )
    exact_eval = None
    if getattr(cfg, "eval_exact", True):
        if true_log_rewards is None:
            true_log_rewards = env._get_states_log_rewards()
        exact_eval = build_exact_evaluator(
            env, true_log_rewards, cfg.eval_rollout_max_length,
            batch_size=getattr(cfg, "eval_policy_batch_size", 1024),
            tolerance=getattr(cfg, "eval_mass_tolerance", 1e-8),
        )

    logger = {
        "tv": [], "tv_samples": [], "sd": [], "l1_empirical": [],
        "traj_length/max": [], "traj_length/mean": [],
        "traj_length/truncated_fraction": [],
    }
    logger.update({f"target/{name}": [value] for name, value in target_metrics.items()})

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
        for name, value in sample_eval(terminal_states).items():
            logger[name].append(value)

        empirical_dist = env.get_empirical_distribution(terminal_states)
        l1_empirical = jnp.abs(true_dist - empirical_dist).sum()
        logger["l1_empirical"].append(l1_empirical)
        if exact_eval is not None:
            terminal_dist, exact_metrics = exact_eval(model_state)
            for name, value in exact_metrics.items():
                logger.setdefault(name, []).append(value)
        else:
            terminal_dist = empirical_dist
            logger["tv"].append(0.5 * l1_empirical)
        logger["traj_length/truncated_fraction"].append(
            jnp.mean(trajectories_length > cfg.eval_rollout_max_length)
        )
        logger["traj_length/max"].append(jnp.max(trajectories_length))
        logger["traj_length/mean"].append(jnp.mean(trajectories_length))
        logger.update(
            env.visualize(
                terminal_dist,
                prefix="terminal_dist",
            )
        )
        return logger

    return short_eval, logger


def prefix_tb_tfbind_trainer(cfg, comet_exp=None):
    key_gen = jax.random.PRNGKey(cfg.seed)

    env: TFBind8Environment = cfg.env
    batch_size = cfg.batch_size
    train_rollout_max_length = cfg.train_rollout_max_length
    eval_rollout_max_length = cfg.eval_rollout_max_length

    key, key_gen = jax.random.split(key_gen)
    model_state = init_model(key, cfg)

    true_log_rewards = env._get_states_log_rewards()
    true_logZ = nn.logsumexp(true_log_rewards)
    true_dist = jnp.exp(true_log_rewards - true_logZ)
    print(f"True logZ: {true_logZ:.4f}")

    initial_dist = env.get_initial_dist()

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
            sequence_length=env.max_length,
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

    eval_fn, logger = get_eval_fn(
        partial(get_eval_rollout_base), env, true_dist, cfg, true_log_rewards
    )
    print(
        f"Target vs target ({batch_size} samples per batch): "
        f"TV samples: {logger['target/tv_samples'][-1]:.4f}, "
        f"SD (Hamming W1): {logger['target/sd'][-1]:.4f}, "
        f"L1 empirical: {logger['target/l1_empirical'][-1]:.4f}"
    )

    def evaluate(model_state, key, step, losses=None):
        eval_key, buffer_key = jax.random.split(key)
        logger.update(eval_fn(model_state, eval_key))

        exact_info = ""
        if "tv_upper_bound" in logger:
            exact_info = (
                f", TV upper: {logger['tv_upper_bound'][-1]:.6f}"
                f", Unabsorbed: {logger['eval/unabsorbed_mass'][-1]:.3e}"
            )
        loss_info = "" if losses is None else f"Loss: {jnp.mean(losses):.4f}, "
        print(
            f"[{step}/{cfg.train_num_steps}] "
            f"{loss_info}"
            f"TV: {logger['tv'][-1]:.4f}, "
            f"TV samples: {logger['tv_samples'][-1]:.4f}, "
            f"SD: {logger['sd'][-1]:.4f}, "
            f"Max Len: {logger['traj_length/max'][-1]:.4f}, "
            f"Mean Len: {logger['traj_length/mean'][-1]:.4f}"
            f"{exact_info}"
        )

        if use_buffer and step > 0:
            buffer_terminal_states, _, _ = buffer.sample(
                buffer_state, buffer_key, batch_size
            )
            buffer_empirical_dist = env.get_empirical_distribution(buffer_terminal_states)
            logger.update(
                env.visualize(buffer_empirical_dist, prefix="buffer_empirical_dist")
            )

        if cfg.use_comet:
            last_entry = extract_last_entry(logger)
            metrics = {}
            for name, value in last_entry.items():
                if isinstance(value, plt.Figure):
                    comet_exp.log_figure(figure=value, figure_name=name, step=step)
                else:
                    metrics[name] = value
            comet_exp.log_metrics(metrics, step=step)

    key, key_gen = jax.random.split(key_gen)
    evaluate(model_state, key, step=0)

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
            grads, (_, _, losses) = loss_bwd_grad_fn(
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
                step=step + 1,
            )

        if ((step + 1) % cfg.eval_frequency == 0) or (step == cfg.train_num_steps - 1):
            key, key_gen = jax.random.split(key_gen)
            evaluate(model_state, key, step=step + 1, losses=losses)

    return model_state, logger
