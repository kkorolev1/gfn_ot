"""The unchanged GFNx AMP policy and portable, numeric checkpoint format."""

import json
import sys
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int

try:
    from gfnx.networks import Encoder
    from gfnx.utils import AMINO_ACIDS, PROTEINS_FULL_ALPHABET
except ModuleNotFoundError as error:
    if error.name != "gfnx":
        raise
    # GFNx's own modules use absolute imports, so load the vendored package
    # under its canonical name when it is not installed in the environment.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    try:
        from gfnx.networks import Encoder
        from gfnx.utils import AMINO_ACIDS, PROTEINS_FULL_ALPHABET
    finally:
        sys.path.pop(0)


class TransformerPolicy(eqx.Module):
    """
    A policy module that uses a simple transformer model to generate
    forward and backward action logits as well as a flow.
    """

    encoder: Encoder
    pooler: eqx.nn.Linear
    train_backward_policy: bool
    n_fwd_actions: int
    n_bwd_actions: int

    def __init__(
        self,
        n_fwd_actions: int,
        n_bwd_actions: int,
        train_backward_policy: bool,
        encoder_params: dict,
        *,
        key: chex.PRNGKey,
    ):
        self.train_backward_policy = train_backward_policy
        self.n_fwd_actions = n_fwd_actions
        self.n_bwd_actions = n_bwd_actions

        output_size = self.n_fwd_actions + 1  # +1 for flow
        if train_backward_policy:
            output_size += n_bwd_actions

        encoder_key, pooler_key = jax.random.split(key)
        self.encoder = Encoder(key=encoder_key, **encoder_params)
        self.pooler = eqx.nn.Linear(
            in_features=encoder_params["hidden_size"],
            out_features=output_size,
            key=pooler_key,
        )

    def __call__(
        self,
        obs_ids: Int[Array, " seq_len"],
        *,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> chex.Array:
        pos_ids = jnp.arange(obs_ids.shape[0])
        encoded_obs = self.encoder(obs_ids, pos_ids, enable_dropout=enable_dropout, key=key)[
            "layers_out"
        ][-1]  # [seq_len, hidden_size]
        encoded_obs = encoded_obs.mean(axis=0)  # Average pooling
        output = self.pooler(encoded_obs)
        if self.train_backward_policy:
            # The TB loss does not use the flow term from the policy.
            # We expect fwd_logits and bwd_logits only.
            # So, we will ignore the flow term here.
            fwd_logits, _, bwd_logits = jnp.split(
                output, [self.n_fwd_actions, self.n_fwd_actions + 1], axis=-1
            )
        else:
            # Similarly, ignore flow if not training backward policy.
            fwd_logits, _ = jnp.split(output, [self.n_fwd_actions], axis=-1)
            bwd_logits = jnp.zeros(shape=(self.n_bwd_actions,), dtype=jnp.float32)
        return {
            "forward_logits": fwd_logits,
            "backward_logits": bwd_logits,
        }


def validate_amp_model(model, max_length):
    """Reject policies whose token or head layout cannot represent AMP."""
    if max_length < 1:
        raise ValueError("AMP sequence length must be positive")
    if model.n_fwd_actions != 21 or model.n_bwd_actions != 1:
        raise ValueError("AMP requires 21 forward actions and one backward action")
    embedder = model.encoder.embedder_block
    if model.encoder.pad_id != 22 or embedder.token_embedder.weight.shape[0] != 23:
        raise ValueError("Model does not use the AMP token alphabet")
    if embedder.position_embedder.pe.shape[0] != max_length + 1:
        raise ValueError("Model positional encoding does not match the AMP sequence length")
    if not model.encoder.layers:
        raise ValueError("AMP Transformer requires at least one encoder layer")
    expected_outputs = 22 + int(model.train_backward_policy)
    if model.pooler.weight.shape[0] != expected_outputs:
        raise ValueError("Model pooler shape does not match the AMP policy heads")


def _encoder_params(model):
    encoder = model.encoder
    embedder = encoder.embedder_block
    layer = encoder.layers[0]
    return {
        "vocab_size": embedder.token_embedder.weight.shape[0],
        "max_length": embedder.position_embedder.pe.shape[0],
        "embedding_size": embedder.token_embedder.weight.shape[1],
        "hidden_size": model.pooler.weight.shape[1],
        "intermediate_size": layer.ff_block.linear.weight.shape[0],
        "num_layers": len(encoder.layers),
        "num_heads": layer.attention_block.num_heads,
        "dropout_rate": float(embedder.position_embedder.dropout.p),
        "attention_dropout_rate": float(layer.attention_block.attention.dropout.p),
        "pad_id": encoder.pad_id,
    }


def save_sampler_checkpoint(path, model: TransformerPolicy, logZ):
    """Export every array and the original architecture without pickle.

    The policy still includes its EOS and optional backward head. Consumers
    choose the fixed-length law by masking EOS and renormalizing at each step.
    Learned logZ is retained for reference; it is not a sequence probability.
    """
    max_length = model.encoder.embedder_block.position_embedder.pe.shape[0] - 1
    validate_amp_model(model, max_length)
    metadata = {
        "format": "gfnx_amp_transformer_v1",
        "max_length": max_length,
        "nchar": len(AMINO_ACIDS),
        "n_fwd_actions": model.n_fwd_actions,
        "n_bwd_actions": model.n_bwd_actions,
        "train_backward_policy": model.train_backward_policy,
        "encoder_params": _encoder_params(model),
        "alphabet": PROTEINS_FULL_ALPHABET,
    }
    template = TransformerPolicy(
        n_fwd_actions=model.n_fwd_actions,
        n_bwd_actions=model.n_bwd_actions,
        train_backward_policy=model.train_backward_policy,
        encoder_params=metadata["encoder_params"],
        key=jax.random.PRNGKey(0),
    )
    params, static = eqx.partition(model, eqx.is_array)
    template_params, template_static = eqx.partition(template, eqx.is_array)
    leaves, structure = jax.tree.flatten(params)
    template_leaves, template_structure = jax.tree.flatten(template_params)
    if structure != template_structure or not eqx.tree_equal(static, template_static):
        raise ValueError("Unsupported AMP Transformer architecture")
    if any(x.shape != y.shape for x, y in zip(leaves, template_leaves)):
        raise ValueError("Unsupported AMP Transformer parameter shapes")
    metadata["num_leaves"] = len(leaves)
    arrays = {
        "metadata": np.asarray(json.dumps(metadata)),
        "logZ": np.asarray(logZ),
        **{f"leaf_{index:04d}": np.asarray(leaf) for index, leaf in enumerate(leaves)},
    }
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez(stream, **arrays)


def load_sampler_checkpoint(path):
    """Restore a complete AMP Transformer; never fill missing weights randomly."""
    with np.load(path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata"]))
        if metadata.get("format") != "gfnx_amp_transformer_v1":
            raise ValueError("Unsupported GFNx AMP sampler checkpoint format")
        if metadata.get("alphabet") != PROTEINS_FULL_ALPHABET or metadata.get("nchar") != 20:
            raise ValueError("Checkpoint does not use the AMP token alphabet")
        model = TransformerPolicy(
            n_fwd_actions=metadata["n_fwd_actions"],
            n_bwd_actions=metadata["n_bwd_actions"],
            train_backward_policy=metadata["train_backward_policy"],
            encoder_params=metadata["encoder_params"],
            key=jax.random.PRNGKey(0),
        )
        validate_amp_model(model, metadata["max_length"])
        params, static = eqx.partition(model, eqx.is_array)
        templates, structure = jax.tree.flatten(params)
        names = [f"leaf_{index:04d}" for index in range(len(templates))]
        if metadata.get("num_leaves") != len(templates) or set(checkpoint.files) != {
            "metadata", "logZ", *names,
        }:
            raise ValueError("Checkpoint does not contain the complete AMP parameter tree")
        leaves = []
        for name, template in zip(names, templates):
            array = checkpoint[name]
            if array.shape != template.shape:
                raise ValueError(f"Invalid shape for {name}: {array.shape}, expected {template.shape}")
            if not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"Checkpoint parameter {name} must be numeric")
            leaves.append(jnp.asarray(array))
        model = eqx.combine(jax.tree.unflatten(structure, leaves), static)
    return model, metadata


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
