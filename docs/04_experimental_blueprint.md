# EGO Experimental Blueprint v1

## Objective
Design experiments that do two jobs at the same time:
1. validate the paper's theoretical claims about stopping and budget sensitivity;
2. demonstrate practical gains over strong heuristic orchestration baselines.

The experiments should be staged from simplest to most realistic.

---

# 1. Core empirical claims to validate

## Claim C1: Better stopping
EGO stops closer to the utility-optimal point than fixed-depth or naive uncertainty heuristics.

## Claim C2: Better cost-performance tradeoff
EGO achieves higher task utility under the same token/latency budget.

## Claim C3: Better action routing
EGO uses tools/delegation more selectively, especially on mixed task distributions.

## Claim C4: Robustness to imperfect uncertainty estimation
Performance degrades gracefully when verifier or posterior estimates are noisy.

These four claims line up directly with Theorems A/B and the algorithm design.

---

# 2. Three-layer evaluation stack

We will use three layers.

## Layer 1: Synthetic oracle environment
Purpose:
- validate theorems under controlled conditions;
- inspect stopping thresholds directly;
- measure utility gaps precisely.

## Layer 2: Semi-synthetic language-agent simulator
Purpose:
- mimic candidate generation, verifier noise, and action-specific information gains;
- retain precise logging and reproducibility.

## Layer 3: Lightweight real tasks
Purpose:
- show that the method still helps on practical agent-style workloads.

This staged design is ideal for AAAI because it connects theory to realistic evidence.

---

# 3. Layer 1: synthetic oracle environment

## 3.1 Hidden state
Each task instance has hidden task type
\[
\tau \in \{\text{reasoning}, \text{retrieval}, \text{math}, \text{code}\}
\]
and hidden answer correctness state summarized by latent confidence
\[
z_t \in [0,1].
\]

The environment starts with initial confidence `z_0` sampled from a task-type-specific distribution.

## 3.2 Action effects
Each action increases confidence stochastically:
\[
z_{t+1} = z_t + \Delta_a(z_t, \tau) + \epsilon_t,
\]
where:
- `\Delta_a(z_t, \tau) >= 0` is the expected information gain of action `a` on task type `\tau`;
- `\Delta_a` decreases with `z_t` to encode diminishing returns;
- `\epsilon_t` is bounded noise.

Example design:
\[
\Delta_a(z_t, \tau) = \rho_{a,\tau}(1-z_t),
\]
with `\rho_{a,\tau}` larger when action `a` matches task type `\tau`.

## 3.3 Observation model
The controller does not observe `z_t` directly. Instead it gets noisy estimated entropy / verifier statistics derived from `z_t`.

Simple mapping:
- oracle entropy proxy `H_t^* = -z_t \log z_t - (1-z_t)\log(1-z_t)`
- observed entropy `\widehat H_t = H_t^* + \eta_t`
- verifier confidence `V_t = z_t + \zeta_t`

This environment is enough to test threshold theorems.

## 3.4 Reward
Stopping at time `T` yields
\[
R_T = \mathbf{1}\{\text{correct}\} - \lambda_{tok} C^{tok}_T - \lambda_{lat} C^{lat}_T - \lambda_{step} T.
\]
The correctness event can be sampled as Bernoulli(`z_T`) or taken as expected correctness `z_T`.

## 3.5 Main plots for Layer 1
- utility vs stopping threshold;
- learned/empirical stopping boundary vs remaining budget;
- action allocation by task type;
- performance under increasing observation noise.

---

# 4. Layer 2: semi-synthetic agent simulator

## 4.1 Motivation
The oracle environment is too clean. We need a simulation that looks like an agent pipeline:
- candidate answers,
- multiple reasoning paths,
- verifier score,
- agreement/disagreement,
- tools and experts with task-dependent usefulness.

## 4.2 Task generation
Construct a pool of instances from templates. Each instance has:
- a hidden correct answer label;
- a task type;
- a difficulty level;
- optional external evidence requirements.

## 4.3 Candidate-generation channels
Each action emits candidate answers according to task-type-specific confusion matrices.

Example:
- on retrieval tasks, `TOOL(search)` has higher probability of producing the correct candidate;
- on math tasks, `DELEGATE(math)` is strongest;
- `THINK` is cheap but improves slowly across all tasks.

This directly tests routing quality.

## 4.4 Verifier model
The verifier scores candidates with some calibration error.
We can control:
- noise level,
- overconfidence bias,
- underconfidence bias,
- type-specific reliability.

This lets us test Theorem B style robustness.

## 4.5 Measured state
From the candidate pool we compute exactly the same `H_t, M_t, D_t, V_t` used by EGO-v1.
Thus Layer 2 is implementation-faithful.

---

# 5. Layer 3: lightweight real-task benchmarks

We should keep this part tractable. The first paper does not need a huge benchmark suite if the synthetic validation is strong.

## Recommended task buckets
1. **Factual QA / retrieval-heavy**
   - questions that benefit from search/tool use
2. **Math / symbolic**
   - questions that benefit from specialized reasoning
3. **Code / execution-mediated**
   - tasks where coding/tool experts help
4. **Mixed routing set**
   - tasks pooled from all three categories

## What matters most
The benchmark must show that a single fixed workflow is suboptimal, making orchestration decisions actually matter.

---

# 6. Baselines

We need baselines that are easy to understand and hard to dismiss.

## B1. Immediate stop
Produce one draft and stop.

## B2. Fixed-k think
Always perform exactly `k` reasoning steps before answering.

## B3. Full-budget think
Use all remaining budget on internal reasoning.

## B4. Tool-first heuristic
Always call tool first when available, then think, then stop.

## B5. Delegate-first heuristic
Always ask an expert first.

## B6. Uncertainty-threshold heuristic
Stop when entropy is below a fixed threshold, independent of remaining budget.

## B7. Cost-unaware VOI router
Choose the action with maximal estimated information gain, ignoring cost.

## B8. Contextual-bandit router without stopping gate
A learned router that decides actions but does not use budget-aware threshold stopping.

These baselines cover the main alternative explanations.

---

# 7. Metrics

## Primary metrics
- task success / accuracy
- expected utility
- total tokens used
- total latency
- number of actions taken

## Secondary metrics
- number of tool calls
- number of delegated calls
- average stopping depth
- stopping regret relative to oracle stop point (synthetic layers)
- calibration error of uncertainty proxy

## Presentation metrics
For the paper, the strongest plots will likely be:
- utility vs budget;
- accuracy vs cost Pareto frontier;
- stopping depth histogram;
- action mix by task type;
- robustness under verifier noise.

---

# 8. Ablations

## A1. Remove budget-aware threshold
Use fixed threshold only.

## A2. Remove value-based action routing
Use the gate, but route actions heuristically.

## A3. Entropy-only
Use only entropy in state and stopping rule.

## A4. No disagreement / no verifier
Remove `D_t` or `V_t`.

## A5. Wrong cost weights
Test sensitivity to misspecified cost tradeoff.

## A6. Posterior noise
Inject controlled noise into candidate scores.

These ablations will show which components are actually responsible for gains.

---

# 9. Milestones

## Milestone M1: proof-aligned synthetic simulator
Deliverable:
- a small simulator implementing hidden confidence state, action-dependent gains, budget, and noisy uncertainty estimates.

Success criterion:
- we can empirically visualize budget-dependent stopping thresholds.

## Milestone M2: EGO-v1 controller in simulator
Deliverable:
- EGO-v1 runs end-to-end and beats fixed-depth baselines on expected utility.

## Milestone M3: robustness study
Deliverable:
- systematic plots under increasing verifier/posterior noise.

## Milestone M4: realistic mixed-task benchmark
Deliverable:
- EGO-v1 outperforms heuristics on at least one mixed routing benchmark.

## Milestone M5: paper figures
Deliverable:
- 4-6 clean figures ready for the draft.

---

# 10. Immediate implementation order

1. implement Layer 1 synthetic oracle environment;
2. implement EGO-v1 threshold + action selection in that environment;
3. test baselines B1-B7 in Layer 1;
4. add posterior/verifier noise and test robustness;
5. only then move to Layer 2 candidate-based simulator;
6. after that, add one lightweight real benchmark.

This order is efficient because it gets us theorem-validation plots early.
