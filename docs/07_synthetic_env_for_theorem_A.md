# Synthetic Environment Aligned with Theorem A

## Purpose
This document defines the first synthetic environment for EGO. Its job is not to be realistic in a language-model sense; its job is to align tightly with Theorem A and empirically validate the entropy-threshold prediction.

This environment should support the following paper claim:
> when uncertainty decreases with diminishing returns and continuation has positive cost, the empirically optimal stopping rule is approximately threshold-based and depends monotonically on remaining budget.

---

# 1. Design principles

The environment should be:
- simple enough that every assumption is transparent;
- cheap to simulate at scale;
- rich enough to compare stopping rules and routing heuristics;
- easy to plot in a paper.

We therefore begin with a scalar latent uncertainty state and only later add agent-style candidate pools.

---

# 2. State and budget

Each episode is defined by:
- latent entropy state `H_t \in [0, \bar H]`;
- remaining step budget `b_t \in \{0,1,\dots,B\}`.

Initial entropy is sampled as
\[
H_0 \sim \text{Uniform}[H_{min}, H_{max}]
\]
or from a task-mixture distribution if we want heterogeneous difficulty.

The episode terminates when the controller chooses `STOP` or when `b_t = 0`.

---

# 3. Action space

For the theorem-validation environment we use:
- `STOP`
- `CONTINUE`

This deliberately matches Theorem A exactly.

In a later extension we can refine `CONTINUE` into several continuation actions, but the first simulator should stay minimal.

---

# 4. Entropy dynamics

## 4.1 Canonical linear-contraction model
When the controller chooses `CONTINUE`, entropy evolves as
\[
H_{t+1} = \max\{0, (1-\rho)H_t + \xi_t\},
\]
where:
- `\rho \in (0,1)` is the information-gain rate;
- `\xi_t` is bounded zero-mean noise, e.g. `\xi_t \sim \text{Unif}[-\sigma,\sigma]`.

This model directly satisfies the intuition that expected entropy shrinks after one more step.

## 4.2 Diminishing returns variant
To make diminishing returns more explicit, we can instead use
\[
H_{t+1} = \max\{0, H_t - \rho H_t^{\gamma} + \xi_t\},
\]
with `\gamma \in (0,1]`.

Interpretation:
- larger entropy states allow larger absolute improvement;
- once entropy is already small, further improvement becomes small.

This model is especially good for illustrating the theorem.

## 4.3 Heterogeneous task variant
For richer experiments, sample a hidden task type `\tau` and let
\[
H_{t+1} = \max\{0, H_t - \rho_{\tau} H_t^{\gamma} + \xi_t\}.
\]
This lets us test whether the threshold phenomenon is robust to task heterogeneity.

---

# 5. Stop reward / utility

We need a stop reward that improves as entropy decreases.
A simple expected-utility model is
\[
G(H_t, b_t) = 1 - \alpha H_t.
\]
This corresponds to expected correctness decreasing linearly with entropy.

A richer model includes explicit budget accounting:
\[
U_T = 1 - \alpha H_T - \lambda_{step}(B - b_T).
\]
Equivalent recursive form:
- stopping at time `t` yields `1 - \alpha H_t`;
- continuing incurs immediate cost `c > 0`.

To align with Theorem A, the cleanest choice is:
\[
\text{STOP payoff} = 1 - \alpha H_t,
\qquad
\text{CONTINUE cost} = c.
\]

---

# 6. Oracle-optimal policy computation

This environment is fully known, so the oracle-optimal policy can be computed by dynamic programming on a discretized entropy grid.

## 6.1 Discretization
Choose grid
\[
\mathcal{H}_{grid} = \{0, \Delta_H, 2\Delta_H, \dots, \bar H\}.
\]
For each remaining budget `b`, compute
\[
V_b(H)
=
\max\{G(H,b), -c + \mathbb{E}[V_{b-1}(H')]\}.
\]

## 6.2 Extract threshold
For each `b`, define the empirical threshold
\[
\hat h^*(b) = \sup \{ H \in \mathcal{H}_{grid} : \text{STOP is optimal at } (H,b) \}.
\]

Then plot `\hat h^*(b)` against `b`.

This is the main theory-validation figure.

---

# 7. Policies to compare

## P1. Oracle dynamic-programming policy
This is the upper benchmark.

## P2. Budget-aware threshold policy
Stop if
\[
H_t \le h_0 + \frac{\beta}{b_t+1}.
\]
This is the EGO-style hand-designed policy.

## P3. Fixed threshold policy
Stop if `H_t \le h_fixed`.
This tests whether budget dependence actually matters.

## P4. Fixed-depth policy
Continue until exactly `k` steps have been used, then stop.

## P5. Never-stop-early policy
Continue until budget is exhausted.

These baselines are enough to support the first result section.

---

# 8. Metrics

## M1. Expected utility
\[
\mathbb{E}[U_T].
\]
This is the most important metric.

## M2. Average stopping time
How many steps are used before stopping.

## M3. Threshold approximation error
Compare the chosen stop rule against oracle thresholds:
\[
\sum_b |\hat h_{policy}(b) - \hat h^*(b)|.
\]

## M4. Stopping regret
Difference between achieved utility and oracle dynamic-programming utility.

## M5. Robustness under entropy-noise corruption
Inject observation noise into entropy estimates:
\[
\widehat H_t = H_t + \eta_t.
\]
Then evaluate degradation.

---

# 9. Main figures to generate

## Figure F1: oracle threshold vs remaining budget
x-axis: remaining budget `b`  
y-axis: optimal threshold `\hat h^*(b)`

Expected trend:
- threshold is weakly decreasing in budget when continuation becomes relatively cheaper with larger remaining budget.

## Figure F2: utility vs initial entropy
For each policy, plot expected utility as a function of `H_0`.

Expected trend:
- fixed-depth policies waste budget at low entropy and under-invest at high entropy.

## Figure F3: utility vs continuation cost
Vary `c` and show how threshold policies adapt.

Expected trend:
- as cost increases, stopping becomes earlier.

## Figure F4: robustness to observation noise
Vary `\sigma_{obs}` in `\widehat H_t = H_t + \eta_t`.

Expected trend:
- budget-aware threshold is more robust than naive fixed-depth baselines.

---

# 10. Parameter suggestions

A reasonable default setting:
- `\bar H = 1.0`
- `B = 8`
- `\rho = 0.25`
- `\gamma = 1.0` or `0.7`
- noise `\xi_t \sim \text{Unif}[-0.03, 0.03]`
- stop utility slope `\alpha = 0.8`
- continuation cost `c = 0.08`

These parameters should create a visible but nontrivial tradeoff.

---

# 11. Why this environment is paper-useful

This synthetic environment does three important things:
1. makes Theorem A visually testable;
2. produces interpretable failure modes for fixed heuristics;
3. gives us a fast sandbox before building richer agent simulators.

It is exactly the right first experiment for a theory-driven AAAI paper.

---

# 12. Immediate implementation checklist

- implement entropy-grid dynamic programming solver;
- implement stochastic episode simulator;
- implement policies P1-P5;
- verify threshold monotonicity numerically;
- create the four core plots;
- save chosen default parameters in code for reproducibility.
