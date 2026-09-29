"""AMP rewards given by a frozen GFlowNet's sequence probability."""

import math
from pathlib import Path

import jax
import jax.numpy as jnp

from .amp_model import AutoregressiveSampler
from gfnx.base import BaseRewardModule
from gfnx.environment.amp import AMPEnvironment


# A  R  N   D  C   E  Q  G  H  I  L  K  M  F  P  S  T  W  Y  V
AA_CHARGES = (0, 1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)


def net_charge(tokens):
    """Sum integer residue charges along the last axis of AMP token arrays."""
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    return jnp.sum(jnp.asarray(AA_CHARGES, dtype=jnp.int32)[tokens], axis=-1)


def validate_gflownet_reward(beta, reward_type):
    if reward_type not in ("power", "charge"):
        raise ValueError(f"Unknown AMP GFlowNet reward type: {reward_type}")
    if not math.isfinite(beta):
        raise ValueError("beta must be finite")
    if reward_type == "power" and beta <= 0:
        raise ValueError("beta must be positive for the power reward")


def gflownet_log_reward(log_p, tokens, beta, reward_type):
    """Shared terminal reward for the acyclic and non-acyclic AMP tasks."""
    if reward_type == "charge":
        return log_p - beta * net_charge(tokens)
    if reward_type == "power":
        return beta * log_p
    raise ValueError(f"Unknown AMP GFlowNet reward type: {reward_type}")


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
    """R(x) = p(x)**beta or p(x) * exp(-beta * net_charge(x)).

    The teacher is frozen, with dropout and EOS disabled. Its saved logZ is
    not a sequence probability. TB consumes log_reward directly, so no tiny
    probabilities are exponentiated, clipped, or renormalized in its target.
    """

    def __init__(self, checkpoint, beta=2.0, reward_type="power"):
        validate_gflownet_reward(beta, reward_type)
        self.checkpoint = resolve_sampler_checkpoint(checkpoint)
        self.beta = float(beta)
        self.reward_type = reward_type
        self.sampler = AutoregressiveSampler.load(self.checkpoint)
        if self.sampler.max_length != 60:
            raise ValueError("The AMP teacher checkpoint must generate length-60 sequences")

    def init(self, rng_key, dummy_state):
        # The teacher stays outside the student's optimizer parameter tree.
        return {}

    def log_reward(self, state, env_params):
        log_p = self.sampler.log_prob(state.tokens)
        return jax.lax.stop_gradient(gflownet_log_reward(log_p, state.tokens, self.beta, self.reward_type))

    def reward(self, state, env_params):
        return jnp.exp(self.log_reward(state, env_params))

    def estimate_log_partition(self, rng_key, num_samples=256, batch_size=16):
        """Importance estimate of log Z_beta, used only to initialize logZ."""
        if num_samples < 1 or batch_size < 1:
            raise ValueError("Partition initialization sample and batch sizes must be positive")
        if (self.reward_type == "power" and self.beta == 1.0) or (
            self.reward_type == "charge" and self.beta == 0.0
        ):
            return jnp.array(0.0)

        @jax.jit
        def sample_log_weights(key):
            tokens = self.sampler.sample(key, (batch_size,))
            if self.reward_type == "charge":
                return -self.beta * net_charge(tokens)
            return (self.beta - 1.0) * self.sampler.log_prob(tokens)

        chunks = []
        for start in range(0, num_samples, batch_size):
            rng_key, sample_key = jax.random.split(rng_key)
            chunks.append(sample_log_weights(sample_key)[:min(batch_size, num_samples - start)])
        log_weights = jnp.concatenate(chunks)
        return jax.scipy.special.logsumexp(log_weights) - jnp.log(num_samples)
