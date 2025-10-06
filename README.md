Improving Credit Assignment with Sequence Models
===========================

Annie Feng — azf@mit.edu

Courtney Ma — cma26@mit.edu

Jinhee Won — jinhee@mit.edu

Irene Terpstra — terpstra@mit.edu

6.8200 Computational Sensorimotor Learning — Final Project (Spring 2024)



Abstract
===========================

We study the long-term credit assignment problem through three lenses: the Decision Transformer (DT), Policy Gradients Incorporating the Future (PGIF), and the Structured State Space Sequence Model (S4). We re-implement the methods, reproduce key results, and evaluate S4 both on its own and when inserted into DT and PGIF. Across standard continuous-control benchmarks and custom environments specifically designed to stress long-horizon reasoning, we observe minimal to modest improvements. We also introduce a Key-To-Door environment to probe DT’s ability to propagate credit over long horizons; DT reliably (though not optimally) solves the task, with performance bounded by the maximum trained positional horizon.

Motivation and Summary of Approach
===========================

Long-term credit assignment remains challenging for both value-based and policy-gradient methods. Recent sequence models promise better long-range dependency handling. We therefore compare S4, which is efficient on very long sequences, to DT and PGIF. Our evaluation spans two Gymnasium tasks (Hopper, Walker) and two custom tasks crafted for credit-assignment stress tests: Key-To-Door and Shortest Paths from Random Walks. Beyond standalone comparisons, we replace DT’s Transformer backbone with S4 and also use S4 as the backward model within PGIF to test whether stronger sequence modeling translates into better credit assignment under fixed compute.

Problem Formulation
===========================

We consider four environments and define their interfaces, signals, and success criteria. For Hopper and Walker, the objective is to move forward rapidly and stably; the action spaces consist of torque commands, and observations contain joint/torso kinematics with standard reward shaping (standing bonus, forward velocity, torque penalty).

The Key-To-Door environment is a three-room gridworld with a key, an empty corridor, and a locked door. The agent must retrieve the key and then traverse to and unlock the door to receive a sparse, terminal reward. The observation includes the agent pose and inventory state; the action set comprises pickup, directional moves, drop, toggle, and termination. This setting concentrates the reward far in time from the initial decisions, making it a useful probe for long-horizon credit assignment.

The Shortest Paths from Random Walks task uses a 20-node graph. DT is trained on random-walk trajectories and asked at test time to produce near-optimal shortest paths from a start to a goal node. We study two dataset variants: one where invalid moves become no-ops (remaining at the same node) and another from a Python “walker” package that disallows no-ops. Both variants reveal how well sequence models can infer long-range structure from suboptimal demonstrations.

Learning Algorithms
===========================

Decision Transformer (DT)

DT treats returns, states, and actions as a sequence and predicts actions autoregressively, conditioned on a desired return-to-go. We re-implement the model, training, and evaluation for all environments. Optimization uses AdamW with a warmup schedule, following paper defaults for Gym tasks and light tuning for our custom tasks.

Structured State Space Sequence Model (S4)

S4 attains strong results on long-range benchmarks while being computationally efficient for long contexts. Given that long-term credit assignment is fundamentally about modeling far-spanning dependencies, we test S4 both standalone and as a drop-in backbone inside DT and as the backward model within PGIF.

Policy Gradients Incorporating the Future (PGIF)

PGIF augments PPO with a backward sequence model that summarizes future information to condition the policy. We implement the method with an LSTM backward pass, latent variables (Z-forcing), and KL regularization to limit over-reliance on the backward network. We also substitute S4 for the backward model to test whether a stronger sequence model yields better credit propagation under the same compute budget.

Overall objective (schematic):

J_PG+ = E_{τ_i ~ D} [ Q_ψ(s_t, a_t, u_t) - α * log π_θ(a_t | s_t, z_t^PG) ]
        + α * J_aux-PG + β * J_KL-PG


Results
===========================

Decision Transformer

On Gym environments, we reproduce the qualitative trends from the DT paper: on Hopper our average reward is close to reported numbers, and on Walker our results slightly exceed the paper’s baseline under our compute budget.

For Key-To-Door, we generate datasets from a random policy and force a large majority of trajectories to be successful. DT reliably solves the task up to a horizon of 4096 steps. However, performance degrades beyond the maximum trained timestep embedding, demonstrating a principal limitation: without an explicit mechanism for extrapolating beyond trained positions, DT’s ability to handle truly long horizons is bounded by its positional range.

For Shortest Paths from Random Walks, DT trained on random-walk trajectories learns to output paths whose length distribution approaches the shortest-path distribution on 2-hop cases. We compare three distributions—shortest path, transformer-generated path, and random walk—to visualize how closely the model imitates optimal planning under different dataset assumptions.

S4 (standalone and integrated)

When trained directly on classification (sCIFAR), our S4 re-implementation reaches strong but slightly below-paper validation accuracy given our limited training budget. In continuous control, S4’s performance is generally comparable to PGIF baselines under equal compute, suggesting that while better sequence modeling helps stability and long-context conditioning, it does not by itself solve long-term credit assignment under the constraints we tested.

PGIF and PGIF + S4

On Hopper-v2, the mean episodic reward steadily increases for both PGIF with a backward LSTM (“PGIF Force”) and PGIF with S4 as the backward model. The two curves track closely for most of training, with late-stage variance favoring different runs at different times.

On Walker2d-v2, PGIF + S4 learns quickly in early training and reaches competitive performance with the LSTM-backed PGIF baseline. The relative advantage is not consistent across seeds in our compute regime.

Overall, integrating S4 into PGIF did not yield a clear or consistent uplift over the baseline in our time and compute settings, although it remained competitive and sometimes learned faster early on.

Discussion
===========================

Our experiments indicate that replacing Transformers with S4 can preserve or slightly improve learning dynamics in some cases, but does not consistently unlock substantially better long-horizon credit assignment under fixed budgets. DT’s success on Key-To-Door shows that sequence models can exploit return conditioning to solve sparse-reward tasks when horizons fit within their positional range. However, DT’s positional embedding ceiling becomes a hard limit: beyond the trained horizon, performance drops, underscoring the need for mechanisms that generalize in time.

References
===========================

Lili Chen et al. Decision Transformer: Reinforcement Learning via Sequence Modeling. arXiv:2106.01345 (2021).

Patrick Coady. Trust Region Policy Optimization with GAE. 2018.

Albert Gu, Karan Goel, Christopher Ré. Efficiently Modeling Long Sequences with Structured State Spaces. arXiv:2111.00396 (2021).

Thomas Mesnard et al. Counterfactual Credit Assignment in Model-Free Reinforcement Learning. arXiv:2011.09464 (2020).

David Venuto et al. Policy Gradients Incorporating the Future. arXiv:2108.02096 (2021).
