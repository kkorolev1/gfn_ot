import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from envs.tfbind.autoregressive import AutoregressiveSampler
from envs.tfbind.tokens import NUCLEOTIDES, NUCLEOTIDES_FULL_ALPHABET


class TFBind8Environment:
    """Fixed-length strings with character replacements and a stop action.

    Action position * nchar + character replaces the selected character.
    Replacements by the current character are masked by the policy. Both
    directions use this encoding, so a reverse action restores the old token.
    """

    def __init__(self, beta=0.1, checkpoint=None, sampler=None):
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.max_length = 8
        self.nchar = len(NUCLEOTIDES)
        self.ntoken = len(NUCLEOTIDES_FULL_ALPHABET)
        self.char_to_id = {char: i for i, char in enumerate(NUCLEOTIDES_FULL_ALPHABET)}
        self.bos_token = self.char_to_id["[BOS]"]
        self.eos_token = self.char_to_id["[EOS]"]
        self.pad_token = self.char_to_id["[PAD]"]
        self.beta = beta
        self.stop_action = self.max_length * self.nchar
        self.num_actions = self.stop_action + 1
        self.sampler = AutoregressiveSampler.load(checkpoint) if checkpoint else sampler
        if self.sampler is not None and (
            self.sampler.max_length != self.max_length
            or self.sampler.nchar != self.nchar
        ):
            raise ValueError("Sampler length/alphabet do not match the environment")

    def get_obs(self, states):
        bos = jnp.full((*states.shape[:-1], 1), self.bos_token, dtype=states.dtype)
        return jnp.concatenate((bos, states), axis=-1)

    def get_initial_dist(self):
        if self.sampler is None:
            raise ValueError(
                "Provide a frozen autoregressive sampler or a compatible sampler checkpoint"
            )
        return self.sampler.sample, self.sampler.log_prob

    def log_initial_reward(self, states):
        _, log_prob = self.get_initial_dist()
        return log_prob(states)

    def log_reward(self, states):
        return self.beta * self.log_initial_reward(states)

    def log_terminal_reward(self, states):
        return self.log_reward(states)

    # Use under vmap
    def step(self, state, action, is_terminal):
        state = state.astype(jnp.int32)
        terminal = is_terminal | (action == self.stop_action)
        state_next = jax.lax.cond(
            terminal,
            lambda: state,
            lambda: state.at[action // self.nchar].set(action % self.nchar),
        )
        return state_next, terminal

    def step_backward(self, state, action, is_terminal):
        return self.step(state, action, is_terminal)

    def get_backward_action(self, state, action):
        position = jnp.minimum(action // self.nchar, self.max_length - 1)
        return jnp.where(
            action == self.stop_action,
            self.stop_action,
            position * self.nchar + state[position].astype(jnp.int32),
        )

    def _get_states_log_rewards(self, batch_size=1024):
        # Chunk enumeration to avoid materializing all MLP activations at once.
        evaluate = jax.jit(self.log_reward)
        chunks = []
        for start in range(0, self.nchar**self.max_length, batch_size):
            indices = jnp.arange(
                start, min(start + batch_size, self.nchar**self.max_length)
            )
            states = jnp.stack(
                jnp.unravel_index(indices, (self.nchar,) * self.max_length), axis=-1
            )
            chunks.append(evaluate(states))
        return jnp.concatenate(chunks).reshape((self.nchar,) * self.max_length)

    def _get_states_rewards(self):
        return jnp.exp(self._get_states_log_rewards())

    @property
    def name(self):
        return "TFBind8-v0"

    @property
    def is_enumerable(self):
        return True

    def get_true_distribution(self):
        log_rewards = self._get_states_log_rewards()
        return jax.nn.softmax(log_rewards.reshape(-1)).reshape(log_rewards.shape)

    def get_empirical_distribution(self, states):
        dist_shape = (self.nchar,) * self.max_length
        indices = jnp.ravel_multi_index(
            states.astype(jnp.int32).T, dist_shape, mode="clip"
        )
        counts = jnp.bincount(indices, length=self.nchar**self.max_length)
        return (counts / counts.sum()).reshape(dist_shape)

    def get_normalizing_constant(self):
        return jnp.exp(jax.nn.logsumexp(self._get_states_log_rewards()))

    def get_ground_truth_sampling(self, key, batch_size):
        log_rewards = self._get_states_log_rewards().reshape(-1)
        indices = jax.random.categorical(key, log_rewards, shape=(batch_size,))
        return jnp.stack(
            jnp.unravel_index(indices, (self.nchar,) * self.max_length), axis=-1
        ).astype(jnp.int32)

    def visualize(self, rewards, prefix="", show=False):
        # Position-wise marginals remain readable for eight-dimensional strings.
        marginals = jnp.stack(
            [
                rewards.sum(axis=tuple(j for j in range(self.max_length) if j != i))
                for i in range(self.max_length)
            ]
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(marginals.T, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xlabel("Position")
        ax.set_ylabel("Character")
        ax.set_yticks(range(self.nchar))
        fig.colorbar(im, ax=ax, label="Marginal mass")
        fig.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)
        return {f"figures/{prefix + '_' if prefix else ''}vis": [fig]}
