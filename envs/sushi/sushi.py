"""Normalized ranking rewards and adjacent-swap dynamics for SUSHI set A."""

from functools import cached_property
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logsumexp


# Set A has different item IDs from set B / sushi3.idata.
SUSHI_ITEMS = (
    "ebi", "anago", "maguro", "ika", "uni", "ikura", "tamago", "toro",
    "tekka_maki", "kappa_maki",
)


class PlackettLuce:
    """Permutations list item IDs from most to least preferred."""

    def __init__(self, log_worth):
        values = np.asarray(log_worth, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("log_worth must be a finite vector with at least two items")
        self.log_worth = jnp.asarray(values - logsumexp(values))
        self.num_items = len(values)

    def log_prob(self, states):
        ordered = self.log_worth[jnp.asarray(states, dtype=jnp.int32)]
        denominators = jax.lax.associative_scan(jnp.logaddexp, ordered, axis=ordered.ndim - 1, reverse=True)
        return jnp.sum(ordered - denominators, axis=-1)

    def sample(self, key, sample_shape=()):
        # Gumbel top-k gives the same law as sequential sampling without replacement.
        scores = self.log_worth + jax.random.gumbel(key, (*sample_shape, self.num_items))
        return jnp.argsort(-scores, axis=-1).astype(jnp.int32)


def kendall_distance(left, right):
    """Number of differently ordered item pairs, i.e. shortest adjacent-swap path."""
    left, right = jnp.asarray(left), jnp.asarray(right)
    left_positions, right_positions = jnp.argsort(left), jnp.argsort(right)
    i, j = np.triu_indices(left.shape[-1], k=1)
    return jnp.sum((left_positions[..., i] < left_positions[..., j]) !=
                   (right_positions[..., i] < right_positions[..., j]), axis=-1)


def plackett_luce_kendall_ot(left, right):
    """Exact Kendall OT: shared Gumbels attain every pairwise mismatch bound.

    For each item pair, both orders threshold the same Gumbel difference.
    The probability of disagreement is therefore |P_L(i<j) - P_R(i<j)|.
    Summing these simultaneously attained lower bounds gives the optimal cost.
    """
    if left.num_items != right.num_items:
        raise ValueError('The two laws must rank the same items')
    i, j = np.triu_indices(left.num_items, k=1)
    theta_left = np.asarray(left.log_worth, dtype=np.float64)
    theta_right = np.asarray(right.log_worth, dtype=np.float64)
    return float(np.abs(expit(theta_left[i] - theta_left[j]) -
                        expit(theta_right[i] - theta_right[j])).sum())


class SushiEnvironment:
    """All permutations of n items; action i swaps positions i and i+1.

    Action n-1 stops. Swaps are their own inverse in both policy directions.
    L and R are normalized Plackett-Luce laws, so Z_L = Z_R = 1.
    """

    def __init__(self, checkpoint=None, left_log_worth=None, right_log_worth=None):
        self.metadata = {}
        if checkpoint is not None:
            path = Path(checkpoint).expanduser()
            if not path.is_absolute() and not path.exists():
                path = Path(__file__).resolve().parents[2] / path
            if not path.exists():
                raise FileNotFoundError(f"Ranking checkpoint {path} not found; run python -m envs.sushi.fit")
            with np.load(path, allow_pickle=False) as data:
                left_log_worth = data['left_log_worth']
                right_log_worth = data['right_log_worth']
                self.metadata = json.loads(str(data['metadata'].item()))
        self.left = PlackettLuce(left_log_worth)
        self.right = PlackettLuce(right_log_worth)
        if self.left.num_items != self.right.num_items:
            raise ValueError("L and R must rank the same number of items")
        self.max_length = self.nchar = self.left.num_items
        if self.max_length > 10:
            raise ValueError("Exact SUSHI evaluation supports at most 10 items")
        self.stop_action = self.max_length - 1
        self.num_actions = self.max_length
        self.num_states = math.factorial(self.max_length)
        self.item_names = self.metadata.get('item_names',
                                           SUSHI_ITEMS if self.max_length == 10 else list(map(str, range(self.max_length))))
        if len(self.item_names) != self.max_length:
            raise ValueError("Checkpoint item_names must match the ranking length")

    @property
    def name(self):
        return "SUSHI-v0"

    @property
    def is_enumerable(self):
        return True

    def get_initial_dist(self):
        return self.left.sample, self.left.log_prob

    def log_initial_reward(self, states):
        return self.left.log_prob(states)

    def log_reward(self, states):
        return self.right.log_prob(states)

    def log_terminal_reward(self, states):
        return self.log_reward(states)

    def step(self, state, action, is_terminal):
        state = state.astype(jnp.int32)
        terminal = is_terminal | (action == self.stop_action)
        state_next = jax.lax.cond(
            terminal, lambda: state,
            lambda: state.at[action].set(state[action + 1]).at[action + 1].set(state[action]),
        )
        return state_next, terminal

    def step_backward(self, state, action, is_terminal):
        return self.step(state, action, is_terminal)

    def get_backward_action(self, state, action):
        return action

    @cached_property
    def all_states(self):
        # Lexicographic unranking via Lehmer codes; 10! x 10 uses 36 MB, not
        # millions of Python tuples or a 10**10 categorical state tensor.
        indices = np.arange(self.num_states)
        states = np.empty((self.num_states, self.max_length), dtype=np.int8)
        for i in range(self.max_length):
            states[:, i] = (indices // math.factorial(self.max_length - i - 1)) % (self.max_length - i)
        for i in range(self.max_length - 2, -1, -1):
            states[:, i + 1:] += states[:, i + 1:] >= states[:, i, None]
        return states

    def get_state_indices(self, states):
        states = np.asarray(states, dtype=np.int32)
        indices = np.zeros(states.shape[:-1], dtype=np.int64)
        for i in range(self.max_length - 1):
            indices += math.factorial(self.max_length - i - 1) * (
                states[..., i + 1:] < states[..., i, None]).sum(axis=-1)
        return indices

    def _get_states_log_rewards(self, batch_size=65536):
        result = np.empty(self.num_states, dtype=np.float64)
        log_worth = np.asarray(self.right.log_worth, dtype=np.float64)
        for start in range(0, self.num_states, batch_size):
            ordered = log_worth[self.all_states[start:start + batch_size]]
            denominator = np.logaddexp.accumulate(ordered[:, ::-1], axis=-1)[:, ::-1]
            result[start:start + batch_size] = (ordered - denominator).sum(axis=-1)
        return result

    def get_true_distribution(self):
        return np.exp(self._get_states_log_rewards())

    def get_normalizing_constant(self):
        return 1.0

    def get_ground_truth_sampling(self, key, batch_size):
        return self.right.sample(key, (batch_size,))

    def get_empirical_distribution(self, states):
        indices = self.get_state_indices(states)
        return np.bincount(indices, minlength=self.num_states) / len(indices)

    def visualize(self, rewards, prefix="", show=False):
        probs = np.asarray(rewards).reshape(-1)
        marginals = np.stack([np.bincount(self.all_states[:, i], weights=probs,
                                          minlength=self.nchar) for i in range(self.max_length)])
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(marginals.T, origin='lower', aspect='auto', cmap='viridis')
        ax.set_xlabel('Preference position (1 = most preferred)')
        ax.set_xticks(range(self.max_length), range(1, self.max_length + 1))
        ax.set_yticks(range(self.nchar), self.item_names)
        fig.colorbar(im, ax=ax, label='Marginal probability')
        fig.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)
        return {f"figures/{prefix + '_' if prefix else ''}vis": [fig]}
