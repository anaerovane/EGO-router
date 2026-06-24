# Prototype Status After First Synthetic Run

## Date
2026-05-22

## What is implemented
The first runnable prototype now includes:
- scalar entropy synthetic environment;
- dynamic-programming oracle solver;
- oracle threshold extraction by remaining budget;
- baseline policies;
- minimal EGO-style entropy-threshold policy;
- end-to-end experiment script.

Implemented files:
- `src/envs/synthetic_entropy_env.py`
- `src/solvers/dp_oracle.py`
- `src/policies/baselines.py`
- `src/policies/ego_threshold.py`
- `scripts/run_synthetic_theorem_a.py`

---

## Important correction made
During implementation, we caught a monotonicity-sign issue in the theorem notes.

If the stopping rule is
\[
\text{STOP when } H \le h^*(b),
\]
then **having more remaining budget should make the controller more willing to continue**, which means the stopping region should shrink, not expand.

Therefore the correct direction is:
\[
h^*(b+1) \le h^*(b).
\]

This correction has been applied to:
- `docs/05_theorem_A_draft.md`
- `docs/06_theorem_A_refined.md`
- `docs/07_synthetic_env_for_theorem_A.md`

---

## First run configuration
The synthetic script currently uses:
- `budget = 8`
- `rho = 0.25`
- `gamma = 0.7`
- `process_noise = 0.03`
- `observation_noise = 0.02`
- `alpha = 0.8`
- `continuation_cost = 0.06`
- `scarcity_cost_scale = 0.08`

The scarcity cost makes continuation effectively more expensive when budget is nearly exhausted, which exposes the budget-dependent threshold effect more clearly.

---

## First oracle thresholds
The dynamic-programming oracle produced the following thresholds:

- `b = 0  ->  h* = 1.0000`
- `b = 1  ->  h* = 0.6000`
- `b = 2  ->  h* = 0.3700`
- `b = 3  ->  h* = 0.3000`
- `b = 4  ->  h* = 0.2700`
- `b = 5  ->  h* = 0.2500`
- `b = 6  ->  h* = 0.2367`
- `b = 7  ->  h* = 0.2267`
- `b = 8  ->  h* = 0.2200`

This is the qualitative pattern we wanted:
- with **more budget**, the stopping threshold is **lower**, meaning the controller is willing to continue until uncertainty becomes very small;
- with **less budget**, the threshold is **higher**, meaning the controller stops earlier.

---

## First policy ranking
The first run produced the following ranking by average reward:

1. `oracle_threshold`: `0.6915`
2. `fixed_threshold_0.25`: `0.6914`
3. `budget_threshold`: `0.6889`
4. `ego_entropy_gate`: `0.6889`
5. `fixed_threshold_0.40`: `0.6759`
6. `fixed_depth_2`: `0.6668`
7. `fixed_depth_4`: `0.6354`
8. `immediate_stop`: `0.5944`
9. `never_stop_early`: `0.2949`

### Initial interpretation
- The prototype behaves sensibly: oracle is best.
- Budget-aware thresholding is competitive and clearly better than fixed-depth heuristics.
- The current task distribution happens to make a tuned fixed threshold surprisingly strong.
- This means the next experimental task is not just "make EGO work", but specifically to identify regimes where **budget awareness** matters clearly.

---

## What this means scientifically
This first run already supports two useful claims:
1. threshold-style stopping is sensible in the theorem-aligned environment;
2. fixed-depth heuristics are clearly suboptimal.

But it does **not yet strongly separate** budget-aware thresholding from the best fixed-threshold baseline.

That is normal for a first prototype.

---

## Immediate next experimental tasks
1. sweep over continuation cost and scarcity-cost scale;
2. sweep over initial entropy distribution;
3. compare fixed-threshold baselines across a wide threshold grid instead of only two hand-picked thresholds;
4. add observation-noise sweep to connect to Theorem B;
5. log per-budget stopping frequencies to compare with oracle thresholds directly.

These are the next steps needed to turn the prototype into paper evidence.
