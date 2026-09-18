import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from libs.gfnx.baselines.utils.tfbind_model import MLPPolicy
from envs.tfbind.tokens import NUCLEOTIDES, NUCLEOTIDES_FULL_ALPHABET


class AutoregressiveSampler:
    def __init__(self, model: MLPPolicy):
        self.model = jax.tree.map(
            lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf,
            model,
        )
        self.max_length = 8
        self.nchar = len(NUCLEOTIDES)
        self.bos_token = NUCLEOTIDES_FULL_ALPHABET.index("[BOS]")
        self.pad_token = NUCLEOTIDES_FULL_ALPHABET.index("[PAD]")
        if model.n_fwd_actions != self.nchar:
            raise ValueError("TFBind8 requires four forward actions")

    def get_obs(self, states):
        bos = jnp.full((*states.shape[:-1], 1), self.bos_token, dtype=states.dtype)
        return jnp.concatenate((bos, states), axis=-1)

    def log_action_probs(self, states):
        """Conditional nucleotide probabilities for a batch of PAD-filled prefixes."""
        obs = self.get_obs(states)
        outputs = jax.vmap(self.model)(obs.reshape((-1, self.max_length + 1)))
        logits = outputs["forward_logits"].reshape((*states.shape[:-1], self.nchar))
        return jax.nn.log_softmax(logits, axis=-1)

    def sample(self, key, sample_shape=()):
        states = jnp.full(
            (*sample_shape, self.max_length), self.pad_token, dtype=jnp.int32
        )

        def add_character(carry, position):
            states, key_gen = carry
            key_gen, key_action = jax.random.split(key_gen)
            token = jax.random.categorical(key_action, self.log_action_probs(states))
            return (
                states.at[..., position].set(token.astype(jnp.int32)),
                key_gen,
            ), None

        (states, _), _ = jax.lax.scan(
            add_character, (states, key), jnp.arange(self.max_length)
        )
        return states

    def log_prob(self, states):
        """Exact log p(x); neither the flow head nor the learned logZ enters p(x)."""
        states = jnp.asarray(states, dtype=jnp.int32)
        if states.ndim == 0 or states.shape[-1] != self.max_length:
            raise ValueError("TFBind8 sequences must have length 8")
        prefix = jnp.full_like(states, self.pad_token)

        def add_character(carry, position):
            prefix, log_p = carry
            log_probs = self.log_action_probs(prefix)
            token = states[..., position]
            log_p += jnp.take_along_axis(log_probs, token[..., None], axis=-1)[..., 0]
            return (prefix.at[..., position].set(token), log_p), None

        (_, log_p), _ = jax.lax.scan(
            add_character,
            (prefix, jnp.zeros(states.shape[:-1])),
            jnp.arange(self.max_length),
        )
        return log_p

    @classmethod
    def load(cls, path):
        """Restore GFNx's Equinox MLPPolicy, with its original weight layout and heads."""
        with np.load(path, allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"]))
            if metadata["format"] != "gfnx_tfbind_mlp_v1":
                raise ValueError("Unsupported GFNx sampler checkpoint format")
            if (
                metadata["max_length"] != 8
                or metadata["nchar"] != 4
                or metadata["alphabet"] != NUCLEOTIDES_FULL_ALPHABET
            ):
                raise ValueError("Checkpoint does not use the TFBind8 token alphabet")
            model = MLPPolicy(
                n_fwd_actions=metadata["nchar"],
                n_bwd_actions=metadata["n_bwd_actions"],
                train_backward_policy=metadata["train_backward_policy"],
                encoder_params={
                    "hidden_size": metadata["num_hid"],
                    "depth": metadata["num_layers"],
                },
                key=jax.random.PRNGKey(0),
            )

            def load_array(name, template):
                array = jnp.asarray(checkpoint[name])
                if array.shape != template.shape:
                    raise ValueError(
                        f"Invalid shape for {name}: {array.shape}, expected {template.shape}"
                    )
                return array

            for i, layer in enumerate(model.encoder.layers):
                model = eqx.tree_at(
                    lambda m: (m.encoder.layers[i].weight, m.encoder.layers[i].bias),
                    model,
                    (
                        load_array(f"encoder_{i}_weight", layer.weight),
                        load_array(f"encoder_{i}_bias", layer.bias),
                    ),
                )
            model = eqx.tree_at(
                lambda m: (m.pooler.weight, m.pooler.bias),
                model,
                (
                    load_array("pooler_weight", model.pooler.weight),
                    load_array("pooler_bias", model.pooler.bias),
                ),
            )
        return cls(model)
