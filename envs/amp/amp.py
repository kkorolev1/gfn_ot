import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from envs.amp.autoregressive import AutoregressiveSampler
from envs.amp.tokens import AMINO_ACIDS, PROTEINS_FULL_ALPHABET
from libs.gfnx.baselines.utils.amp_reward import gflownet_log_reward, validate_gflownet_reward


class AMPEnvironment:
    """Length-60 proteins with single amino-acid replacements and a stop action.

    L(x)=p(x); R(x)=p(x)**beta or p(x)*exp(-beta*net_charge(x)).
    BOS, EOS and PAD retain GFNx's IDs but are never editable residues.
    """

    def __init__(self, beta=2.0, checkpoint=None, sampler=None, reward_type="power"):
        validate_gflownet_reward(beta, reward_type)
        self.max_length = 60
        self.nchar = len(AMINO_ACIDS)
        self.ntoken = len(PROTEINS_FULL_ALPHABET)
        self.char_to_id = {char: i for i, char in enumerate(PROTEINS_FULL_ALPHABET)}
        self.bos_token = self.char_to_id["[BOS]"]
        self.eos_token = self.char_to_id["[EOS]"]
        self.pad_token = self.char_to_id["[PAD]"]
        self.beta = beta
        self.reward_type = reward_type
        self.stop_action = self.max_length * self.nchar
        self.num_actions = self.stop_action + 1
        self.sampler = AutoregressiveSampler.load(checkpoint) if checkpoint else sampler
        if self.sampler is not None and (
            self.sampler.max_length != self.max_length or self.sampler.nchar != self.nchar
        ):
            raise ValueError("Sampler length/alphabet do not match the AMP environment")

    def get_obs(self, states):
        last_token = states[..., -1]
        to_append = jnp.where(
            (last_token == self.pad_token) | (last_token == self.eos_token),
            self.pad_token, self.eos_token,
        )
        return jnp.concatenate((states, to_append[..., None]), axis=-1)

    def get_initial_dist(self):
        if self.sampler is None:
            raise ValueError("Provide a frozen AMP sampler or a GFNx AMP sampler checkpoint")
        return self.sampler.sample, self.sampler.log_prob

    def log_initial_reward(self, states):
        _, log_prob = self.get_initial_dist()
        return log_prob(states)

    def log_reward(self, states):
        log_p = self.log_initial_reward(states)
        return gflownet_log_reward(log_p, states, self.beta, self.reward_type)

    def log_terminal_reward(self, states):
        return self.log_reward(states)

    # Use under vmap, as in TFBind.
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
            action == self.stop_action, self.stop_action,
            position * self.nchar + state[position].astype(jnp.int32),
        )

    @property
    def name(self):
        return "AMP-v0"

    @property
    def is_enumerable(self):
        return False

    def get_position_marginals(self, states):
        """Per-position amino-acid frequencies, not a joint distribution."""
        return jax.nn.one_hot(states.astype(jnp.int32), self.nchar).mean(axis=0)

    def visualize(self, marginals, prefix="", show=False):
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(marginals.T, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xlabel("Position")
        ax.set_ylabel("Amino acid")
        ax.set_yticks(range(self.nchar), AMINO_ACIDS)
        fig.colorbar(im, ax=ax, label="Marginal mass")
        fig.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)
        return {f"figures/{prefix + '_' if prefix else ''}vis": [fig]}
