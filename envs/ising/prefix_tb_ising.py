from functools import partial
import math

import flax.linen as flax_nn
from flax.training.train_state import TrainState
from flax.traverse_util import path_aware_map
import jax
import jax.numpy as jnp
import jax.nn as nn
import numpy as np
import optax

from envs.ising.ising import IsingEnvironment
from envs.ising.metrics import (
    correlation_error,
    correlation_profile,
    magnetization_error,
    sinkhorn_distance,
)
from envs.ising.reference import get_reference_pair, sample_reference
from envs.tfbind.buffer import build_terminal_state_buffer
from utils.helper import log1mexp


class Model(flax_nn.Module):
    max_length: int = 256
    nchar: int = 2
    num_layers: int = 2
    num_hid: int = 64

    weight_init: float = 1e-8
    bias_init: float = 1e-1

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
        mask = jnp.concatenate(
            (encoded.astype(bool), jnp.zeros((*states.shape[:-1], 1), dtype=bool)),
            axis=-1,
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
    init_logZ = 0.0 if cfg.init_logZ is None else cfg.init_logZ
    params["params"] = {**params["params"], "logZ": jnp.array((init_logZ,))}
    optimizers_map = {
        "model_optim": optax.adam(learning_rate=build_lr_schedule(cfg.lr)),
        "logZ_optim": (
            optax.set_to_zero()
            if cfg.logZ_lr == 0
            else optax.adam(learning_rate=build_lr_schedule(cfg.logZ_lr))
        ),
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
    env: IsingEnvironment,
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


def sample_initial_states(key, env, batch_size, initial_buffer=None):
    if initial_buffer is None:
        return env.sample_uniform(key, (batch_size,))
    indices = jax.random.randint(key, (batch_size,), 0, len(initial_buffer))
    return initial_buffer[indices]


def get_train_rollout(
    key_gen,
    model_state,
    params,
    env: IsingEnvironment,
    batch_size: int,
    rollout_max_length: int,
    is_forward_rollout: bool = True,
    terminal_states=None,
    terminal_log_rewards=None,
    initial_buffer=None,
):
    if is_forward_rollout:
        key, key_gen = jax.random.split(key_gen)
        states = sample_initial_states(key, env, batch_size, initial_buffer)
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
    log_pfs = jnp.concatenate(
        [env.log_initial_reward(init_states)[:, None], log_pfs], axis=1
    )
    _, log_pb_initial, _ = model_state.apply_fn(
        params, jax.lax.stop_gradient(init_states), log_rewards[:, 0]
    )
    log_pbs = jnp.concatenate([log_pb_initial[..., -1][:, None], log_pbs], axis=1)
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
    if reg_coef != 0:
        flow_penalties = jnp.exp(reg_coef + log_flows) * weights
    else:
        flow_penalties = jnp.zeros_like(tb_losses)

    losses = tb_losses.sum(-1) + flow_penalties.sum(-1)
    return jnp.mean(losses), (
        terminal_states,
        log_rewards[:, -1],
        jax.lax.stop_gradient(losses),
        jax.lax.stop_gradient(tb_losses),
        jax.lax.stop_gradient(flow_penalties),
    )


def get_path_rollout(
    key, model_state, params, states, env, rollout_max_length, is_forward=True
):
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
    log_weights = (
        pb[:, -1] - env.log_initial_reward(origin)
        if is_forward
        else env.log_reward(origin) - pf[:, -1]
    )
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
        return (
            next_states,
            next_done,
            key_gen,
            weights,
            lengths + (~done),
            iteration + 1,
        )

    initial = (
        states,
        jnp.zeros(len(states), dtype=bool),
        key,
        log_weights,
        jnp.zeros(len(states), dtype=jnp.int32),
        jnp.array(0),
    )
    final, done, _, weights, lengths, _ = jax.lax.while_loop(cond, step, initial)
    return {
        "initial_states": origin if is_forward else final,
        "terminal_states": final if is_forward else origin,
        "log_weights": jnp.where(done, weights, jnp.nan),
        "lengths": jnp.where(done, lengths, rollout_max_length + 1),
        "completed": done,
    }


def get_eval_fn(get_eval_rollout_fn, env, true_dist, cfg):
    del get_eval_rollout_fn, true_dist
    left_reference, right_reference = get_reference_pair(env, cfg.reference, cfg.seed)
    left_pool, right_pool = jnp.asarray(left_reference), jnp.asarray(right_reference)
    forward_rollout = jax.jit(
        partial(
            get_path_rollout,
            env=env,
            rollout_max_length=cfg.eval_rollout_max_length,
            is_forward=True,
        )
    )
    backward_rollout = jax.jit(
        partial(
            get_path_rollout,
            env=env,
            rollout_max_length=cfg.eval_rollout_max_length,
            is_forward=False,
        )
    )
    bound_offset = 0.0 if cfg.source_logZ is None else float(cfg.source_logZ)
    metric_names = (
        "elbo",
        "eubo",
        "sinkhorn",
        "magnetization",
        "correlation",
        "sinkhorn_converged",
        "sinkhorn_error",
        "eval/num_completed",
        "eval/num_backward_completed",
        "eval/sinkhorn_num_samples",
        "traj_length/max",
        "traj_length/mean",
        "traj_length/truncated_fraction",
        "backward_traj_length/max",
        "backward_traj_length/mean",
        "backward_traj_length/truncated_fraction",
    )
    logger = {name: [] for name in metric_names}
    logger["data/target_correlations"] = [
        correlation_profile(right_reference, env.lattice_size)
    ]

    def short_eval(model_state, key):
        source_key, target_key, fwd_key, bwd_key, subset_key = jax.random.split(key, 5)
        starts = left_pool[
            jax.random.randint(source_key, (cfg.eval_batch_size,), 0, len(left_pool))
        ]
        targets = right_pool[
            jax.random.randint(target_key, (cfg.eval_batch_size,), 0, len(right_pool))
        ]
        forward = forward_rollout(fwd_key, model_state, model_state.params, starts)
        backward = backward_rollout(bwd_key, model_state, model_state.params, targets)
        completed = np.asarray(forward["completed"])
        bwd_completed = np.asarray(backward["completed"])
        terminal_states = np.asarray(forward["terminal_states"])
        samples = terminal_states[completed]
        values = {
            name: float("nan")
            for name in (
                "elbo",
                "eubo",
                "sinkhorn",
                "magnetization",
                "correlation",
                "sinkhorn_error",
            )
        }
        values.update(
            {
                "sinkhorn_converged": False,
                "eval/num_completed": int(completed.sum()),
                "eval/num_backward_completed": int(bwd_completed.sum()),
                "eval/sinkhorn_num_samples": 0,
            }
        )
        # Conditional averages after dropping unfinished paths are not evidence
        # bounds. Leave a gap in the metric until every path has terminated.
        if np.all(completed):
            values["elbo"] = (
                float(np.asarray(forward["log_weights"], dtype=np.float64).mean())
                + bound_offset
            )
        if np.all(bwd_completed):
            values["eubo"] = (
                float(np.asarray(backward["log_weights"], dtype=np.float64).mean())
                + bound_offset
            )
        if len(samples):
            values["magnetization"] = magnetization_error(samples, env.lattice_size)
            values["correlation"] = correlation_error(
                samples, right_reference, env.lattice_size
            )
            count = min(len(samples), cfg.sinkhorn.sample_size, len(targets))
            indices = np.asarray(jax.random.permutation(subset_key, len(samples)))[
                :count
            ]
            values.update(
                sinkhorn_distance(
                    samples[indices],
                    np.asarray(targets)[:count],
                    epsilon=cfg.sinkhorn.epsilon,
                    threshold=cfg.sinkhorn.threshold,
                    max_iterations=cfg.sinkhorn.max_iterations,
                )
            )
            values["eval/sinkhorn_num_samples"] = count
            logger.update(env.visualize(samples, prefix="terminal_states"))
            logger["data/model_correlations"] = [
                correlation_profile(samples, env.lattice_size)
            ]
        else:
            logger["figures/terminal_states_vis"] = []
            logger["data/model_correlations"] = []
        for prefix, paths in (
            ("traj_length", forward),
            ("backward_traj_length", backward),
        ):
            lengths = np.asarray(paths["lengths"])
            values[f"{prefix}/max"] = int(lengths.max())
            values[f"{prefix}/mean"] = float(lengths.mean())
            values[f"{prefix}/truncated_fraction"] = float(
                1 - np.asarray(paths["completed"]).mean()
            )
        for name, value in values.items():
            logger[name].append(value)
        logger["data/terminal_states"] = [terminal_states]
        logger["data/initial_states"] = [np.asarray(starts)]
        logger["data/trajectory_lengths"] = [np.asarray(forward["lengths"])]
        logger["data/backward_trajectory_lengths"] = [np.asarray(backward["lengths"])]
        logger["data/forward_log_weights"] = [np.asarray(forward["log_weights"])]
        logger["data/backward_log_weights"] = [np.asarray(backward["log_weights"])]
        return logger

    return short_eval, logger


def prefix_tb_ising_trainer(cfg, experiment_logger):
    env = cfg.env
    if cfg.model.max_length != env.max_length or cfg.model.nchar != 2:
        raise ValueError(
            "Ising model.max_length must equal env.lattice_size**2 and model.nchar must be 2"
        )
    if (
        min(
            cfg.batch_size,
            cfg.eval_batch_size,
            cfg.train_rollout_max_length,
            cfg.eval_rollout_max_length,
            cfg.sinkhorn.sample_size,
        )
        < 1
    ):
        raise ValueError(
            "Batch sizes, rollout lengths and Sinkhorn sample size must be positive"
        )
    if cfg.source_logZ is not None and not math.isfinite(cfg.source_logZ):
        raise ValueError("source_logZ must be finite or null")
    if cfg.initial_state_source not in ("uniform", "buffer"):
        raise ValueError("initial_state_source must be uniform or buffer")
    if cfg.initial_state_source == "buffer":
        settings = cfg.initial_buffer
        if (
            min(
                settings.num_samples,
                settings.num_chains,
                settings.thinning,
                settings.update_frequency,
            )
            < 1
            or settings.burn_in < 0
        ):
            raise ValueError(
                "Invalid initial buffer size, MCMC settings or update frequency"
            )
    if cfg.init_logZ is None:
        # Ground-state approximation for the ratio; neither source samples nor
        # a uniform-proposal estimate of an exponentially concentrated Z is used.
        cfg.init_logZ = (
            2 * env.max_length * env.coupling * (env.beta_right - env.beta_left)
        )
    flow_penalty_log_scale = -2 * env.max_length * env.coupling * env.beta_right
    experiment_logger.log_metrics(
        {
            "logZ_ratio_initial": cfg.init_logZ,
            "flow_penalty_log_scale": flow_penalty_log_scale,
        },
        step=0,
    )
    key_gen = jax.random.PRNGKey(cfg.seed)

    batch_size = cfg.batch_size
    train_rollout_max_length = cfg.train_rollout_max_length

    key, key_gen = jax.random.split(key_gen)
    model_state = init_model(key, cfg)

    initial_rng = np.random.default_rng(cfg.seed + 1731)

    def fill_initial_buffer():
        settings = cfg.initial_buffer
        samples = sample_reference(
            env,
            env.beta_left,
            settings.num_samples,
            seed=int(initial_rng.integers(0, 2**32)),
            num_chains=settings.num_chains,
            burn_in=settings.burn_in,
            thinning=settings.thinning,
        )
        return jnp.asarray(samples, dtype=jnp.int32)

    initial_buffer = (
        fill_initial_buffer() if cfg.initial_state_source == "buffer" else None
    )

    get_train_rollout_base = partial(
        get_train_rollout,
        env=env,
        batch_size=batch_size,
        rollout_max_length=train_rollout_max_length,
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
        prefix_tb_loss_fn,
        reg_coef=cfg.reg_coef,
        use_weights=cfg.use_weights,
    )

    @partial(jax.jit)
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_fwd_grad_fn(key, model_state, params, initial_buffer):
        get_train_forward_rollout = partial(
            get_train_rollout_base,
            is_forward_rollout=True,
            initial_buffer=initial_buffer,
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

    eval_fn, logger = get_eval_fn(None, env, None, cfg)

    def evaluate(model_state, key, step, losses=None):
        eval_key, buffer_key = jax.random.split(key)
        logger.update(eval_fn(model_state, eval_key))
        experiment_logger.save_checkpoint(model_state)

        loss_info = "" if losses is None else f"Loss: {jnp.mean(losses):.4f}, "
        metrics_info = "".join(
            f"{label}: {logger[name][-1]:.4f}, "
            for name, label in (
                ("elbo", "ELBO"),
                ("eubo", "EUBO"),
                ("sinkhorn", "Sinkhorn"),
                ("magnetization", "Mag"),
                ("correlation", "Corr"),
            )
            if logger.get(name)
        )
        print(
            f"[{step}/{cfg.train_num_steps}] "
            f"{loss_info}"
            f"{metrics_info}"
            f"Max Len: {logger['traj_length/max'][-1]:.4f}, "
            f"Mean Len: {logger['traj_length/mean'][-1]:.4f}"
        )

        if use_buffer and step > 0:
            buffer_terminal_states, _, _ = buffer.sample(
                buffer_state, buffer_key, batch_size
            )
            buffer_empirical_dist = env.get_position_marginals(buffer_terminal_states)
            buffer_name = "buffer_position_marginals"
            logger[f"data/{buffer_name}"] = [buffer_empirical_dist]
            logger.update(env.visualize(buffer_empirical_dist, prefix=buffer_name))

        experiment_logger.log_evaluation(logger, step=step)

    key, key_gen = jax.random.split(key_gen)
    evaluate(model_state, key, step=0)

    for step in range(cfg.train_num_steps):
        if (
            initial_buffer is not None
            and step > 0
            and step % cfg.initial_buffer.update_frequency == 0
        ):
            initial_buffer = fill_initial_buffer()
        if not use_buffer or step % (buffer_cfg.replay_ratio + 1) == 0:
            key, key_gen = jax.random.split(key_gen)
            grads, (
                terminal_states,
                terminal_log_rewards,
                losses,
                tb_losses,
                flow_penalties,
            ) = loss_fwd_grad_fn(key, model_state, model_state.params, initial_buffer)
            model_state = model_state.apply_gradients(grads=grads)

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
            grads, (_, _, losses, tb_losses, flow_penalties) = loss_bwd_grad_fn(
                key,
                model_state,
                model_state.params,
                terminal_states,
                terminal_log_rewards,
            )
            model_state = model_state.apply_gradients(grads=grads)

        experiment_logger.log_metrics(
            {
                "loss": jnp.mean(losses),
                "tb_loss": jnp.mean(tb_losses),
                "flow_penalties": jnp.mean(flow_penalties),
                "logZ_learned": model_state.params["params"]["logZ"],
            },
            step=step + 1,
        )

        if ((step + 1) % cfg.eval_frequency == 0) or (step == cfg.train_num_steps - 1):
            key, key_gen = jax.random.split(key_gen)
            evaluate(model_state, key, step=step + 1, losses=losses)

    return model_state, logger
