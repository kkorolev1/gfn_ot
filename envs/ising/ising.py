import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


class IsingEnvironment:
    """Periodic, zero-field Ising bridge with binary tokens and single-spin edits.

    Spins are 2 * tokens - 1. L and R are unnormalized Ising weights.
    The prefix-TB logZ is log(Z_R/Z_L).
    """

    def __init__(self, lattice_size=16, beta_left=0.6, beta_right=1.2, coupling=1.0):
        if not isinstance(lattice_size, int) or lattice_size < 2:
            raise ValueError("lattice_size must be an integer of at least 2")
        if any(not math.isfinite(beta) or beta < 0 for beta in (beta_left, beta_right)):
            raise ValueError("Ising inverse temperatures must be finite and nonnegative")
        if not math.isfinite(coupling) or coupling <= 0:
            raise ValueError("coupling must be finite and positive")
        self.lattice_size = lattice_size
        self.max_length = lattice_size**2
        self.nchar = 2
        self.beta_left = beta_left
        self.beta_right = beta_right
        self.coupling = coupling
        self.stop_action = self.max_length * self.nchar
        self.num_actions = self.stop_action + 1

    def energy(self, states):
        states = jnp.asarray(states, dtype=jnp.int32)
        spins = (2 * states - 1).reshape((*states.shape[:-1], self.lattice_size, self.lattice_size))
        neighbors = jnp.roll(spins, -1, axis=-1) + jnp.roll(spins, -1, axis=-2)
        return -self.coupling * jnp.sum(spins * neighbors, axis=(-2, -1))

    def log_initial_reward(self, states):
        return -self.beta_left * self.energy(states)

    def log_reward(self, states):
        return -self.beta_right * self.energy(states)

    def log_terminal_reward(self, states):
        return self.log_reward(states)

    def sample_uniform(self, key, sample_shape=()):
        return jax.random.randint(key, (*sample_shape, self.max_length), 0, 2, dtype=jnp.int32)

    def get_initial_dist(self):
        # The second function supplies source weights to TB, not proposal density.
        # Its missing normalizer is absorbed in the learned logZ ratio.
        return self.sample_uniform, self.log_initial_reward

    def step(self, state, action, is_terminal):
        state = state.astype(jnp.int32)
        terminal = is_terminal | (action == self.stop_action)
        next_state = jax.lax.cond(
            terminal, lambda: state,
            lambda: state.at[action // self.nchar].set(action % self.nchar),
        )
        return next_state, terminal

    def step_backward(self, state, action, is_terminal):
        return self.step(state, action, is_terminal)

    def get_backward_action(self, state, action):
        position = jnp.minimum(action // self.nchar, self.max_length - 1)
        return jnp.where(action == self.stop_action, self.stop_action,
                         position * self.nchar + state[position].astype(jnp.int32))

    @property
    def name(self):
        return f"Ising-{self.lattice_size}x{self.lattice_size}-v0"

    @property
    def is_enumerable(self):
        # Evaluation uses MCMC references, including on small test lattices.
        return False

    def get_position_marginals(self, states):
        return jax.nn.one_hot(states.astype(jnp.int32), 2).mean(axis=0)

    def visualize(self, values, prefix="", show=False):
        values = np.asarray(values)
        if values.shape == (self.max_length, 2):
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(values[:, 1].reshape(self.lattice_size, self.lattice_size), vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, label="P(spin = +1)")
        else:
            count = min(len(values), 16)
            fig, axes = plt.subplots(max(1, (count + 3) // 4), 4, squeeze=False, figsize=(8, 2 * max(1, (count + 3) // 4)))
            for i, ax in enumerate(axes.flat):
                if i < count:
                    ax.imshow(values[i].reshape(self.lattice_size, self.lattice_size), cmap="coolwarm", vmin=0, vmax=1)
                ax.axis("off")
        fig.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)
        return {f"figures/{prefix + '_' if prefix else ''}vis": [fig]}
