"""Ising metrics from arXiv:2602.05961, Appendix C.1.3."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry.epsilon_scheduler import Epsilon
from ott.geometry.geometry import Geometry
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.acceleration import Momentum
from ott.solvers.linear.sinkhorn import Sinkhorn


def _spins(samples, lattice_size):
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] != lattice_size**2:
        raise ValueError("Expected a nonempty batch of flattened Ising lattices")
    if not np.all((samples == 0) | (samples == 1)):
        raise ValueError("Ising metrics expect binary tokens")
    return (2.0 * samples - 1).reshape(-1, lattice_size, lattice_size)


def magnetization_error(samples, lattice_size):
    # At zero external field, the true finite-lattice mean spin is exactly zero.
    means = _spins(samples, lattice_size).mean(axis=0)
    return float((np.abs(means.mean(axis=0)).sum() + np.abs(means.mean(axis=1)).sum())
                 / (2 * lattice_size))


def correlation_profile(samples, lattice_size):
    """Connected correlations, averaged across sites, for both axes and all shifts."""
    spins = _spins(samples, lattice_size)
    means = spins.mean(axis=0)
    return np.asarray([
        [np.mean(spins * np.roll(spins, -r, axis=axis))
         - np.mean(means * np.roll(means, -r, axis=axis - 1))
         for r in range(lattice_size)]
        for axis in (1, 2)
    ])


def correlation_error(samples, reference_samples, lattice_size):
    # Paper denominator is 2L. The public repository uses 4L instead.
    return float(np.abs(correlation_profile(samples, lattice_size)
                        - correlation_profile(reference_samples, lattice_size)).mean())


def sinkhorn_distance(samples, reference_samples, epsilon=.001, threshold=1e-6,
                      max_iterations=10000):
    """Transport cost <plan, Hamming>, with convergence checked at target epsilon.

    This is the paper's Sinkhorn distance, not a debiased Sinkhorn divergence.
    Duplicate compression preserves empirical weights exactly. Float64 and
    epsilon continuation avoid the reference solver's float32/small-epsilon issue.
    """
    samples, reference_samples = np.asarray(samples), np.asarray(reference_samples)
    if (samples.ndim != 2 or reference_samples.ndim != 2
            or samples.shape[1] != reference_samples.shape[1]
            or min(len(samples), len(reference_samples)) < 1):
        raise ValueError("Sinkhorn expects two nonempty batches of equal-length sequences")
    if not np.all((samples == 0) | (samples == 1)) or not np.all((reference_samples == 0) | (reference_samples == 1)):
        raise ValueError("Ising Sinkhorn expects binary tokens")
    if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(threshold) or threshold <= 0 or max_iterations < 1:
        raise ValueError("Invalid Sinkhorn epsilon, threshold or iteration count")
    x, counts_x = np.unique(samples, axis=0, return_counts=True)
    y, counts_y = np.unique(reference_samples, axis=0, return_counts=True)
    # Binary Hamming distance without a (n, m, lattice_size**2) tensor.
    x, y = x.astype(np.float64), y.astype(np.float64)
    cost = x.sum(-1)[:, None] + y.sum(-1)[None, :] - 2 * x @ y.T
    # At cold temperatures the two modes can be 256 flips apart. Starting at
    # epsilon=1 can freeze incorrect masses before the dual potentials balance.
    # Start at the actual cost scale; the final epsilon remains absolute.
    initial_scale = max(1.0, float(cost.max()) / epsilon)
    warmup = int(math.ceil(math.log(initial_scale) / -math.log(.95)))
    with jax.experimental.enable_x64():
        a = jnp.asarray(counts_x / counts_x.sum(), dtype=jnp.float64)
        b = jnp.asarray(counts_y / counts_y.sum(), dtype=jnp.float64)
        geometry = Geometry(cost_matrix=jnp.asarray(cost), scale_cost=1.0,
                            epsilon=Epsilon(target=epsilon, init=initial_scale, decay=.95))
        # Start adaptive overrelaxation after epsilon reaches its target. Applying
        # strong momentum during continuation can stall even identical measures.
        # OTT checks one marginal internally; use a tighter internal tolerance
        # before independently checking both marginals against `threshold`.
        momentum_start = 10 * math.ceil((warmup + 100) / 10)
        result = Sinkhorn(lse_mode=True, threshold=threshold * .1,
                          momentum=Momentum(start=momentum_start, error_threshold=.1),
                          min_iterations=min(warmup + 20, max_iterations),
                          max_iterations=max_iterations)(LinearProblem(geometry, a=a, b=b))
        plan = result.matrix
        error = float(jnp.maximum(jnp.abs(plan.sum(1) - a).sum(),
                                   jnp.abs(plan.sum(0) - b).sum()))
        value = float(jnp.sum(plan * geometry.cost_matrix))
        converged = bool(result.converged) and max_iterations > warmup and error <= threshold and math.isfinite(value)
    return {'sinkhorn': value if converged else float('nan'),
            'sinkhorn_converged': converged, 'sinkhorn_error': error}
