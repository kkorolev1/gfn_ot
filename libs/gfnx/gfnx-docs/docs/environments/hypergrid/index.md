# Hypergrid Environment

The hypergrid environment is a classic GFlowNet benchmark: you build a point on
a $D$-dimensional grid by incrementing coordinates one step at a time and then
decide when to stop. The terminal state is the final grid cell you land on, and
its reward is given by a user-chosen hypergrid reward module. Because every
transition is discrete and the search space is finite, the environment is great
for debugging objectives and policies before moving on to more complicated
domains.

## Intuition

- **State**: a vector of length `dim`, with each coordinate in
  `[0, side - 1]`. The all-zero vector is the initial (empty) state.
- **Actions**: choose a coordinate to increment or issue a *stop* action.
  Coordinates that already reached `side - 1` are automatically clamped, so
  the agent cannot walk off the grid.
- **Trajectory**: a sequence of increments that ends with the stop action.
- **Reward**: determined by the selected hypergrid reward module; higher values
  make the corresponding grid cells more likely under an ideal GFlowNet.

## Key parameters

- `dim`: number of dimensions (default `4`). Increasing it makes the grid grow
  exponentially; keep this small for quick experiments.
- `side`: number of discrete positions per dimension (default `20`).
- `reward_module`: controls how terminal states are scored. The library ships
  with ready-made options described below.

## Quick start

```python
import jax
import jax.numpy as jnp
import gfnx

# 1. Pick a reward; EasyHypergrid is a gentle default.
reward = gfnx.EasyHypergridRewardModule()

# 2. Create the environment.
env = gfnx.HypergridEnvironment(reward_module=reward, dim=3, side=5)
params = env.init(jax.random.PRNGKey(0))

# 3. Reset to get the initial observation/state batch.
obs, state = env.reset(num_envs=1, env_params=params)

# 4. Take a forward step (increment coordinate 0).
action = jnp.array([0], dtype=jnp.int32)
obs, state, log_reward, done, _ = env.step(state, action, params)

print("Terminal?", bool(state.is_terminal[0]))
print("Reward (log scale):", float(log_reward[0]))

# 5. Stop when you are ready to terminate the trajectory. Stop action is typically the last action.
stop = jnp.array([env.action_space.n - 1], dtype=jnp.int32)
obs, state, log_reward, done, _ = env.step(state, stop, params)

print("Terminal?", bool(state.is_terminal[0]))
print("Reward (log scale):", float(log_reward[0]))
```

The environment is vectorised: set `num_envs > 1` and pass batched actions to
interact with multiple trajectories in parallel. `log_reward` is only non-zero
the moment you transition into a terminal state.

## Reward options

- By default the reward assigned to a terminal state $s = (s^1, \ldots, s^D)$ with side length
  `side = H` follows

  $$
  \mathcal{R}(s) = R_0
  + R_1 \prod_{i=1}^D \mathbb{I}\left[0.25 < \left|\frac{s^i}{H-1}-0.5\right|\right]
  + R_2 \prod_{i=1}^D \mathbb{I}\left[0.3 < \left|\tfrac{s^i}{H-1}-0.5\right| < 0.4\right].
  $$

  The indicator products carve out $2^D$ symmetric modes: instead of peaking at the grid centre,
  the reward places mass on annuli that sit away from the middle, making exploration highly
  multimodal. Adjusting $(R_0, R_1, R_2)$ changes how prominent each ring of modes is.

- `gfnx.EasyHypergridRewardModule()` – baseline reward with a gentle peak near
  the center of the mode.
- `gfnx.HardHypergridRewardModule()` – sharper peaks that make exploration and
  credit assignment harder.
- `gfnx.GeneralHypergridRewardModule(R0, R1, R2)` – customise how wide the
  reward plateaus are by tuning the coefficients.

You can plug your own reward by subclassing `BaseRewardModule` or by taking the
general module and adjusting the parameters to match your experiment.

## Exploring the grid

The hypergrid exposes utilities that are handy for analysis and evaluation:

```python
# Enumerate the full state space (dim * side entries).
all_states = env.get_all_states(params)

# Compute the exact partition function and reward-proportional distribution.
Z = env.get_normalizing_constant(params)
true_dist = env.get_true_distribution(params)  # shape = (side,)*dim

# Draw samples directly from the ground-truth distribution.
gt_state = env.get_ground_truth_sampling(
    rng_key=jax.random.PRNGKey(1),
    batch_size=4,
    env_params=params,
)
```

These helpers are invaluable for sanity checks (does your policy match the true
distribution?) and for tracking metrics such as the mean reward or KL divergence
to ground truth. For a deeper dive into the environment and reward APIs, check
the companion pages in this section of the docs.

## API references:

- [Environment](environment_api.md)
- [Reward module](reward_api.md)

## References

- Bengio, E. *et&nbsp;al.* (2021). *Flow network based generative models for non-iterative diverse candidate generation.* 
  Advances in Neural Information Processing Systems (NeurIPS).