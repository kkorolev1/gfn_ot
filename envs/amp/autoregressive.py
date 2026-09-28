"""Frozen GFNx AMP policy restricted to a fixed number of amino acids."""

import equinox as eqx
import jax
import jax.numpy as jnp

from libs.gfnx.baselines.utils.amp_model import (
    TransformerPolicy,
    load_sampler_checkpoint,
    validate_amp_model,
)
from envs.amp.tokens import AMINO_ACIDS, PROTEINS_FULL_ALPHABET


class AutoregressiveSampler:
    """The product of EOS-masked, renormalized forward conditionals.

    This is not the variable-length policy globally conditioned on length 60.
    At each position EOS is removed and the 20 amino-acid logits are normalized.
    Dropout is disabled, and neither the flow head nor learned logZ enters p(x).
    """

    def __init__(self, model: TransformerPolicy, max_length=60):
        validate_amp_model(model, max_length)
        self.model = jax.tree.map(
            lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf,
            model,
        )
        self.max_length = max_length
        self.nchar = len(AMINO_ACIDS)
        self.bos_token = PROTEINS_FULL_ALPHABET.index("[BOS]")
        self.eos_token = PROTEINS_FULL_ALPHABET.index("[EOS]")
        self.pad_token = PROTEINS_FULL_ALPHABET.index("[PAD]")

    def get_obs(self, states):
        """Use GFNx AMP's trailing EOS/PAD observation, with no prepended BOS."""
        states = jnp.asarray(states, dtype=jnp.int32)
        if states.ndim == 0 or states.shape[-1] != self.max_length:
            raise ValueError(f"AMP sequences must have length {self.max_length}")
        last = states[..., -1]
        trailing = jnp.where(
            (last == self.pad_token) | (last == self.eos_token),
            self.pad_token,
            self.eos_token,
        )
        return jnp.concatenate((states, trailing[..., None]), axis=-1)

    def log_action_probs(self, states):
        """Return (..., 20) log probabilities for PAD-filled prefixes."""
        obs = self.get_obs(states)
        outputs = jax.vmap(lambda row: self.model(row, enable_dropout=False))(
            obs.reshape((-1, self.max_length + 1))
        )
        logits = outputs["forward_logits"][..., :self.nchar]
        logits = logits.reshape((*obs.shape[:-1], self.nchar))
        return jax.nn.log_softmax(logits, axis=-1)

    def sample(self, key, sample_shape=()):
        states = jnp.full(
            (*sample_shape, self.max_length), self.pad_token, dtype=jnp.int32
        )

        def add_character(carry, position):
            prefix, key_gen = carry
            key_gen, key_action = jax.random.split(key_gen)
            token = jax.random.categorical(key_action, self.log_action_probs(prefix))
            return (prefix.at[..., position].set(token.astype(jnp.int32)), key_gen), None

        (states, _), _ = jax.lax.scan(
            add_character, (states, key), jnp.arange(self.max_length)
        )
        return states

    def log_prob(self, states):
        """Exact log p(x) for full amino-acid sequences with arbitrary batch dims."""
        states = jnp.asarray(states, dtype=jnp.int32)
        if states.ndim == 0 or states.shape[-1] != self.max_length:
            raise ValueError(f"AMP sequences must have length {self.max_length}")
        prefix = jnp.full_like(states, self.pad_token)

        def add_character(carry, position):
            prefix, log_p = carry
            token = states[..., position]
            log_p += jnp.take_along_axis(
                self.log_action_probs(prefix), token[..., None], axis=-1
            )[..., 0]
            return (prefix.at[..., position].set(token), log_p), None

        (_, log_p), _ = jax.lax.scan(
            add_character,
            (prefix, jnp.zeros(states.shape[:-1])),
            jnp.arange(self.max_length),
        )
        return log_p

    @classmethod
    def load(cls, path):
        model, metadata = load_sampler_checkpoint(path)
        return cls(model, max_length=metadata["max_length"])
