"""AMP rewards given by a frozen GFlowNet's sequence probability."""

import math
from pathlib import Path

import jax
import jax.numpy as jnp

from .amp_model import AutoregressiveSampler
from gfnx.base import BaseRewardModule
from gfnx.environment.amp import AMPEnvironment


def resolve_sampler_checkpoint(checkpoint):
    """Accept absolute, working-directory-relative, or parent-project paths."""
    if not checkpoint:
        raise ValueError("Set environment.sampler_checkpoint to a frozen AMP sampler.npz")
    path = Path(checkpoint).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path(__file__).resolve().parents[4] / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate.absolute()) for candidate in candidates)
    raise FileNotFoundError(f"AMP sampler checkpoint not found. Looked in: {searched}")


class FixedLengthAMPEnvironment(AMPEnvironment):
    """Disable EOS even when the installed GFNx still allows variable lengths."""

    def get_invalid_mask(self, state, env_params):
        return super().get_invalid_mask(state, env_params).at[:, self.stop_action].set(True)


class GFlowNetAMPRewardModule(BaseRewardModule):
    """R(x) = p(x)**beta for the same fixed-length p used by the cyclic task.

    The teacher is frozen, with dropout and EOS disabled. Its saved logZ is
    not a sequence probability. TB consumes log_reward directly, so no tiny
    probabilities are exponentiated, clipped, or renormalized in its target.
    """

    def __init__(self, checkpoint, beta=2.0):
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        self.checkpoint = resolve_sampler_checkpoint(checkpoint)
        self.beta = float(beta)
        self.sampler = AutoregressiveSampler.load(self.checkpoint)
        if self.sampler.max_length != 60:
            raise ValueError("The AMP teacher checkpoint must generate length-60 sequences")

    def init(self, rng_key, dummy_state):
        # The teacher stays outside the student's optimizer parameter tree.
        return {}

    def log_reward(self, state, env_params):
        return jax.lax.stop_gradient(self.beta * self.sampler.log_prob(state.tokens))

    def reward(self, state, env_params):
        return jnp.exp(self.log_reward(state, env_params))

    def estimate_log_partition(self, rng_key, num_samples=256, batch_size=16):
        """Importance estimate of log Z_beta, used only to initialize logZ."""
        if num_samples < 1 or batch_size < 1:
            raise ValueError("Partition initialization sample and batch sizes must be positive")
        if self.beta == 1.0:
            return jnp.array(0.0)

        @jax.jit
        def sample_log_weights(key):
            tokens = self.sampler.sample(key, (batch_size,))
            return (self.beta - 1.0) * self.sampler.log_prob(tokens)

        chunks = []
        for start in range(0, num_samples, batch_size):
            rng_key, sample_key = jax.random.split(rng_key)
            chunks.append(sample_log_weights(sample_key)[:min(batch_size, num_samples - start)])
        log_weights = jnp.concatenate(chunks)
        return jax.scipy.special.logsumexp(log_weights) - jnp.log(num_samples)
