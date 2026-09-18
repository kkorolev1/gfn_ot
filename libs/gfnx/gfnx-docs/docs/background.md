# Background

Generative Flow Networks (GFlowNets) are probabilistic models designed to sample complex combinatorial objects with probability proportional to a user-defined reward. Concretely, we work with a finite discrete space $\mathcal{X}$ and an unnormalized reward function ${\mathcal{R} : \mathcal{X} \rightarrow \mathbb{R}_{\ge 0}}$. Our goal is to learn a sampler whose marginal distribution over objects $x \in \mathcal{X}$ matches $\mathcal{R}(x)/Z$, where the normalizing constant ${Z = \sum_{x \in \mathcal{X}} \mathcal{R}(x)}$ is typically unknown.

To describe the generative process, GFlowNets build a directed acyclic graph (DAG) $\mathcal{G} = (\mathcal{S}, \mathcal{E})$. Every node in $\mathcal{S}$ represents a partially constructed object, edges in $\mathcal{E}$ correspond to incremental edits, and there is a unique initial state $s_0$ with no parents. Terminal states coincide with the target space $\mathcal{X}$, so any complete trajectory $\tau = (s_0, s_1, \ldots, s_{n_\tau})$ records the sequence of construction steps that leads to a finished object $s_{n_\tau} \in \mathcal{X}$.

GFlowNet training revolves around two stochastic policies. The forward policy $\mathcal{P}_F(s' \mid s)$ chooses how to expand a state by selecting one of its children, while the backward policy $\mathcal{P}_B(s' \mid s)$ picks a parent to undo a step. A pair of forward/backward policies is considered proper if the induced distributions over complete trajectories match,

$$
\prod_{t=1}^{n_\tau} \mathcal{P}_F(s_t \mid s_{t-1}) = \frac{\mathcal{R}(s_{n_\tau})}{Z} \prod_{t=1}^{n_\tau} \mathcal{P}_B(s_{t-1} \mid s_t) \quad \text{for all } \tau.
$$

When this *trajectory balance* condition holds, sampling in the forward direction yields terminal states with the desired reward-proportional probabilities.

In practice we parameterize $\mathcal{P}_F$ (and sometimes supporting quantities such as state flows or $\mathcal{P}_B$) with neural networks, then optimize an objective that encourages trajectory balance. The main families of objectives are:

- **Detailed Balance (DB)** enforces local consistency between pairs of neighbouring states by matching forward and backward flows.
- **Trajectory Balance (TB)** lifts the constraint to entire trajectories, introducing a learnable estimate $Z_\theta$ for the normalizing constant.
- **Subtrajectory Balance (SubTB)** interpolates between DB and TB by applying weighted DB constraints to every subpath within a trajectory.

These objectives can be trained on-policy with trajectories sampled from the current forward policy, or off-policy using replay buffers and exploration strategies. Depending on the environment, we may jointly learn $\mathcal{P}_B$ or keep it fixed (e.g., uniform over parents); in either case, there always exists a unique forward policy that satisfies the trajectory balance constraint for a given backward policy. 

The `gfnx` library provides ready-to-use implementations of these ideas, making it straightforward to prototype new objectives, environments, and inference procedures.
