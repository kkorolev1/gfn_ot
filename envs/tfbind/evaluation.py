"""Exact terminal-law evaluation and sample metrics for cyclic TFBind."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment


def sample_metrics(samples, target_samples):
    """Empirical TV and Wasserstein-1 with nucleotide Hamming transport cost.

    Both batches have equal size and uniform sample weights. The optimal
    assignment therefore gives exact empirical W1, measured in replacements.
    Duplicate strings retain their multiplicity in both metrics.
    """
    samples = np.asarray(samples)
    target_samples = np.asarray(target_samples)
    if samples.ndim != 2 or samples.shape != target_samples.shape or len(samples) == 0:
        raise ValueError("Sample batches must have the same nonempty (batch, length) shape")
    batch_size = len(samples)
    _, indices = np.unique(
        np.concatenate((samples, target_samples)), axis=0, return_inverse=True
    )
    counts = np.bincount(indices[:batch_size], minlength=indices.max() + 1)
    target_counts = np.bincount(indices[batch_size:], minlength=len(counts))
    tv = 0.5 * np.abs(counts - target_counts).sum() / batch_size

    costs = (samples[:, None, :] != target_samples[None, :, :]).sum(axis=-1)
    rows, columns = linear_sum_assignment(costs)
    return {"tv_samples": float(tv), "sd": float(costs[rows, columns].mean())}


def build_sample_evaluator(target_dist, batch_size, seed):
    """Draw two independent target batches; reuse the first as eval reference.

    Sampling is with replacement from the normalized p(x)**beta distribution.
    The target-to-target metrics are finite-sample baselines, not zero-distance
    population values. Their batch size matches the policy evaluation batch.
    """
    target_dist = np.asarray(target_dist, dtype=np.float64)
    probs = target_dist.reshape(-1)
    probs = probs / probs.sum()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(probs), size=(2, batch_size), p=probs)
    samples = np.stack(np.unravel_index(indices, target_dist.shape), axis=-1)
    reference, comparison = samples
    baseline = sample_metrics(comparison, reference)
    empirical = np.bincount(indices[1], minlength=len(probs)) / batch_size
    baseline["l1_empirical"] = float(np.abs(empirical - probs).sum())

    def evaluate(samples):
        return sample_metrics(samples, reference)

    return evaluate, baseline


@partial(jax.jit, static_argnames=("max_steps", "tolerance"))
def _propagate(initial_dist, forward_probs, max_steps, tolerance):
    nchar = initial_dist.shape[0]
    max_length = initial_dist.ndim
    stop_probs = forward_probs[:, -1].reshape(initial_dist.shape)
    replacements = forward_probs[:, :-1].reshape(-1, max_length, nchar)
    kernels = tuple(
        replacements[:, position, :].reshape(
            nchar**position, nchar, nchar ** (max_length - position - 1), nchar
        )
        for position in range(max_length)
    )

    def cond(carry):
        step, active, _ = carry
        return (step < max_steps) & (active.sum() > tolerance)

    def advance(carry):
        step, active, stopped = carry
        stopped = stopped + active * stop_probs
        next_active = jnp.zeros_like(active)
        for position, kernel in enumerate(kernels):
            mass = active.reshape(
                nchar**position, nchar, nchar ** (max_length - position - 1)
            )
            # Sum over the old character a; b is the replacement character.
            incoming = jnp.einsum("lar,larb->lbr", mass, kernel)
            next_active = next_active + incoming.reshape(active.shape)
        return step + 1, next_active, stopped

    steps, active, stopped = jax.lax.while_loop(
        cond, advance, (0, initial_dist, jnp.zeros_like(initial_dist))
    )
    # Match evaluation's forced stop at its horizon. If we stopped early at the
    # tolerance, the resulting terminal distribution has TV error <= active.sum().
    return stopped + active, active.sum(), steps


def terminal_distribution(initial_dist, forward_probs, max_steps, tolerance=1e-8):
    """Return capped terminal probabilities, unabsorbed mass, and steps used.

    initial_dist has shape (nchar,) * max_length. Action i*nchar+c replaces
    position i with character c; the final action stops. Forward probabilities
    must include stop and sum to one per state. The residual also bounds the
    TV difference from the eventual terminal law (assuming eventual stopping).
    """
    if max_steps < 0 or tolerance < 0:
        raise ValueError("max_steps and tolerance must be nonnegative")
    # Long cyclic rollouts amplify float32 roundoff. Use float64 only here;
    # the policy and training retain their normal precision.
    with jax.experimental.enable_x64():
        result = _propagate(
            jnp.asarray(initial_dist, dtype=jnp.float64),
            jnp.asarray(forward_probs, dtype=jnp.float64),
            max_steps=max_steps,
            tolerance=tolerance,
        )
    dist, remaining, steps = jax.device_get(result)
    return dist, float(remaining), int(steps)


def build_exact_evaluator(env, true_log_rewards, rollout_max_length, batch_size=1024,
                          tolerance=1e-8):
    """Cache the state space and fixed source/target distributions once."""
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be positive")
    shape = (env.nchar,) * env.max_length
    indices = np.arange(env.nchar**env.max_length)
    states = jnp.asarray(np.stack(np.unravel_index(indices, shape), axis=-1))
    log_rewards = np.asarray(true_log_rewards, dtype=np.float64).reshape(-1)

    def normalize(log_values):
        values = np.exp(log_values - log_values.max())
        return (values / values.sum()).reshape(shape)

    initial_dist = normalize(log_rewards / env.beta)
    target_dist = normalize(log_rewards)

    @jax.jit
    def evaluate_policy(model_state, states):
        log_pf, _, _ = model_state.apply_fn(model_state.params, states)
        return log_pf

    def evaluate(model_state):
        log_probs = np.concatenate([
            np.asarray(evaluate_policy(model_state, states[start:start + batch_size]),
                       dtype=np.float64)
            for start in range(0, len(states), batch_size)
        ])
        probs = np.exp(log_probs - log_probs.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        terminal_dist, residual, steps = terminal_distribution(
            initial_dist, probs, rollout_max_length, tolerance
        )
        tv = 0.5 * np.abs(terminal_dist - target_dist).sum()
        metrics = {
            "tv": float(tv),
            "tv_lower_bound": float(max(0.0, tv - residual)),
            "tv_upper_bound": float(min(1.0, tv + residual)),
            "eval/unabsorbed_mass": residual,
            "eval/propagation_steps": steps,
            "eval/mass_error": float(abs(terminal_dist.sum() - 1.0)),
        }
        return terminal_dist, metrics

    return evaluate
