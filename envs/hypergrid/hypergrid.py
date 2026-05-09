import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


class Hypergrid:
    """
    Hypergrid environment
    """

    def __init__(self, dim: int = 2, side: int = 10) -> None:
        super().__init__()
        self.dim = dim
        self.side = side

        self.stop_action = 2 * self.dim  # Stop action id

    def log_reward(self, state: jnp.ndarray):
        ax = jnp.abs(state / (self.side - 1) - 0.5)
        reward = (
            (ax > 0.25).prod(-1) * 0.5 + ((ax < 0.4) * (ax > 0.3)).prod(-1) * 2 + 1e-3
        )
        return jnp.log(reward)

    # Use under vmap
    def step(self, state: jnp.ndarray, action: jnp.ndarray, is_terminal: jnp.ndarray):
        def get_next_state_terminal():
            return jnp.array(state, copy=True), jnp.array(True)

        def get_next_state_nonterminal():
            state_next = jax.lax.cond(
                (action >= 0) & (action < self.dim),
                lambda: state.at[action].add(-1),
                lambda: state.at[action - self.dim].add(1),
            )
            return state_next, jnp.array(False)

        def get_next_state_from_nonterminal():
            is_terminal_next = action == self.stop_action
            return jax.lax.cond(
                is_terminal_next, get_next_state_terminal, get_next_state_nonterminal
            )

        return jax.lax.cond(
            is_terminal, get_next_state_terminal, get_next_state_from_nonterminal
        )

    def step_backward(
        self, state: jnp.ndarray, action: jnp.ndarray, is_terminal: jnp.ndarray
    ):
        def get_next_state_terminal():
            return jnp.array(state, copy=True), jnp.array(True)

        def get_next_state_nonterminal():
            state_next = jax.lax.cond(
                (action >= 0) & (action < self.dim),
                lambda: state.at[action].add(1),
                lambda: state.at[action - self.dim].add(-1),
            )
            return state_next, jnp.array(False)

        def get_next_state_from_nonterminal():
            is_terminal_next = action == self.stop_action
            return jax.lax.cond(
                is_terminal_next, get_next_state_terminal, get_next_state_nonterminal
            )

        return jax.lax.cond(
            is_terminal, get_next_state_terminal, get_next_state_from_nonterminal
        )

    def get_grid_rewards(self):
        rewards = jnp.zeros((self.side,) * self.dim, dtype=jnp.float32)

        def update_rewards(idx: int, rewards):
            multi_index = jnp.unravel_index(idx, shape=rewards.shape)
            state = jnp.asarray(multi_index, dtype=jnp.float32)[None, ...]
            return rewards.at[multi_index].set(jnp.exp(self.log_reward(state)[0]))

        return jax.lax.fori_loop(0, self.side**self.dim, update_rewards, rewards)

    def visualize(self, rewards, dims=(0, 1), prefix="", show=False):
        assert self.dim == 2, "Visualization is not implemented for d != 2"
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(rewards, origin="lower", cmap="viridis")
        ax.set_xlabel(f"Dim {dims[0]+1}")
        ax.set_ylabel(f"Dim {dims[1]+1}")
        fig.colorbar(im, ax=ax, label="Reward", fraction=0.046, pad=0.04)
        plt.tight_layout()
        figure_dict = {f"figures/{prefix + '_' if prefix else ''}vis": [fig]}
        if show:
            plt.show()
        else:
            plt.close()
        return figure_dict


if __name__ == "__main__":
    env = Hypergrid(2, 10)
    env.visualize(env.get_grid_rewards(), show=True)
