# Correlation Metrics

Correlation metrics quantify how well the learned flows (log ratios recovered via
backward rollouts) align with the environment reward landscape (log rewards of
terminal states). High correlation means the sampler places probability mass in
rewarded regions; low correlation suggests misalignment or mode collapse.

## Intuition

- Practical even when the exact terminal distribution is not enumerable.
- Can be invalid when distinct states share identical rewards (ties destroy
    rank information).
- Batch size should respect accelerator memory; terminal states are reshaped to
    `(n_rounds, batch_size, ...)` for backward rollouts.
- A custom `transform_fn` can project complex terminal states (e.g. sequences,
    grids) into a lower-dimensional representation or bins before applying the
    correlation tests.

## Key parameters

### On-Policy (`OnPolicyCorrelationMetricsModule`)

- `n_rounds`: number of backward-rollout repetitions (averaging reduces variance);
- `n_terminal_states`: total number of terminal states to generate (must be a multiple of `batch_size`);
- `batch_size`: trajectories per rollout batch; trades throughput for memory;
- `fwd_policy_fn` / `bwd_policy_fn`: policy callables returning logits and auxiliary dict with both forward & backward logits;
- `transform_fn`: optional projector applied to states & log values before correlation;
- `env`: environment instance supplying rollout primitives.

### Test-Set (`TestCorrelationMetricsModule`)

- `n_rounds`: number of backward-rollout repetitions applied to the fixed test set;
- `test_set`: batch of terminal states kept constant (tree structure must match environment states);
- `bwd_policy_fn`: backward policy callable returning logits and auxiliary dict;
- `batch_size`: trajectories per backward rollout batch when computing log-ratios;
- `transform_fn`: optional projector for states & log values;
- `env`: environment instance.

## Quick start

See the dedicated variant pages for runnable examples:

- [On-Policy Correlation](on_policy.md) — samples terminal states from the current forward policy at evaluation time.
- [Test-Set Correlation](test_set.md) — reuses a fixed batch of terminal states drawn from the ground-truth distribution.

Both variants return `pearson` (linear correlation) and `spearman` (rank correlation) between transformed log ratios and transformed log rewards.
