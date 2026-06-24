# Theorem B Draft: Utility Loss Under Uncertainty Estimation Error

## Purpose
Theorem A justifies threshold stopping when the controller knows the relevant uncertainty statistic. But EGO does not observe oracle entropy; it estimates uncertainty from finite candidate pools and verifier scores.

Therefore the second main theorem should answer:
> if entropy/confidence estimates are imperfect, how much utility do we lose relative to an oracle stopper?

This theorem is important because it turns calibration from an engineering detail into a theoretical quantity.

---

# 1. Setup

Let:
- `H_t^*` denote the oracle uncertainty statistic at time `t`;
- `\widehat H_t` denote the controller's estimate;
- `h^*(b)` denote the oracle stopping threshold from Theorem A.

We compare two stopping rules:

## Oracle policy
\[
\pi^*(H_t^*, b_t) = \mathbf{1}\{H_t^* > h^*(b_t)\}
\]
where `1` means continue and `0` means stop.

## Estimated-threshold policy
\[
\widehat \pi(\widehat H_t, b_t) = \mathbf{1}\{\widehat H_t > h^*(b_t)\}.
\]

The only difference is that one uses `H_t^*` and the other uses `\widehat H_t`.

---

# 2. Error model

Assume the uncertainty estimator is uniformly accurate with high probability.

## Assumption E1: bounded estimation error
For all times `t \le T`,
\[
|\widehat H_t - H_t^*| \le \varepsilon
\]
with probability at least `1-\delta`.

This can be derived from concentration of candidate-vote statistics or bounded verifier-score perturbations in later sections.

## Assumption E2: local smoothness of continuation advantage
Let
\[
\Delta_b(H) := Q_{cont}(H,b) - Q_{stop}(H,b)
\]
be the continuation advantage from Theorem A.
Assume `\Delta_b(H)` is `L`-Lipschitz in `H` for each `b`.

## Assumption E3: bounded utility range
The total episode utility lies in `[U_{min}, U_{max}]`.
For convenience let `U_{range} := U_{max} - U_{min}`.

---

# 3. Intuition before theorem

If the estimate error is at most `\varepsilon`, the estimated policy can only disagree with the oracle policy near the threshold `h^*(b)`.

Far away from the threshold, both policies choose the same action.

So utility loss should be controlled by:
1. how often the process visits states within an `\varepsilon`-tube around the threshold;
2. how sensitive the continuation advantage is to entropy perturbations.

This is the key idea.

---

# 4. First theorem statement: generic worst-case bound

## Theorem B1 (worst-case utility gap)
Suppose Assumptions E1-E3 hold and the horizon is at most `T`. Then the utility gap between the oracle policy and the estimated-threshold policy satisfies
\[
\mathbb{E}[U_T(\pi^*) - U_T(\widehat \pi)]
\le
L T \varepsilon + U_{range} T \delta.
\]

### Interpretation
- `L T \varepsilon`: loss due to small but systematic uncertainty-estimation error;
- `U_{range} T \delta`: loss due to rare estimator failures.

This is a coarse but very clean statement.

---

# 5. Sharper theorem: margin-dependent bound

The worst-case bound above is easy to state but may be loose. A better version uses the threshold margin.

Define the threshold margin at state `(H,b)` as
\[
m_b(H) := |H - h^*(b)|.
\]

If `m_b(H) > \varepsilon`, then the oracle and estimated policies must agree.
So disagreement occurs only when
\[
m_b(H_t^*) \le \varepsilon.
\]

## Theorem B2 (margin-dependent disagreement bound)
Under Assumptions E1-E3,
\[
\mathbb{E}[U_T(\pi^*) - U_T(\widehat \pi)]
\le
L \sum_{t=1}^T \mathbb{P}(m_{b_t}(H_t^*) \le \varepsilon)
+ U_{range} T \delta.
\]

### Why this is stronger
This theorem says estimation error matters only near the decision boundary. If the process usually lies far from the threshold, then estimated stopping is nearly oracle-optimal even with moderate estimator noise.

This is a very nice message for the paper.

---

# 6. Proof sketch for Theorem B1

## Step 1: couple the two policies
Construct a coupling where both policies evolve on the same randomness, differing only in whether they stop or continue at a given decision point.

## Step 2: identify disagreement events
Under E1, if `|H_t^* - h^*(b_t)| > \varepsilon`, then oracle and estimated policies agree.
Disagreement is only possible inside the threshold tube.

## Step 3: bound per-step decision error
By Lipschitz continuity of `\Delta_b(H)`, mis-evaluating the threshold by `\varepsilon` changes the continuation-minus-stop advantage by at most `L\varepsilon`.
So the utility impact of a mistaken threshold comparison at one step is at most `L\varepsilon` on the good event.

## Step 4: sum over the horizon
Across at most `T` decision points, cumulative loss is at most `LT\varepsilon` on the good event.
Add the estimator-failure event contribution `U_{range}T\delta`.

This yields the result.

---

# 7. How to connect this theorem to EGO-v1

EGO-v1 does not estimate entropy directly from an oracle state. It constructs a posterior from candidate answers and verifier scores:
\[
p_t(y) \propto \exp(S_t(y)/\tau),
\qquad
\widehat H_t = -\sum_y p_t(y) \log p_t(y).
\]

So in the final paper, Theorem B should be paired with a short proposition of the form:

## Proposition B0 (entropy perturbation from score perturbation)
If candidate scores satisfy
\[
\max_y |\widehat S_t(y) - S_t^*(y)| \le \eta,
\]
then the induced entropy estimate satisfies
\[
|\widehat H_t - H_t^*| \le C \eta
\]
for a constant `C` depending on temperature and candidate-set size.

Then Theorem B applies with `\varepsilon = C\eta`.

This gives a clean route from score calibration to stopping robustness.

---

# 8. Why this theorem matters for AAAI

This theorem does several good things for the paper:
- it elevates uncertainty estimation from heuristic confidence scoring to a mathematically relevant quantity;
- it explains why adding verifier and disagreement signals can improve stopping robustness;
- it gives a principled reason to report calibration/error analyses in experiments.

In other words, it ties theory and empirical protocol together.

---

# 9. What to validate empirically

Theorem B suggests a direct synthetic experiment:

1. compute oracle entropy `H_t^*` in the simulator;
2. generate noisy estimates `\widehat H_t = H_t^* + \eta_t`;
3. compare oracle stopping and noisy stopping;
4. plot utility gap vs noise scale `\sigma`.

Expected trend:
- approximately linear degradation for small noise;
- stronger robustness when most states are far from the threshold.

This could become one of the cleanest plots in the paper.

---

# 10. Recommended final positioning

For the main paper, Theorem B should probably be stated in the margin-dependent form because it is both sharper and more intuitive.

A concise main-text version:

> When the controller uses an estimated uncertainty statistic instead of oracle uncertainty, stopping disagreements occur only near the budget-dependent decision boundary. Under bounded estimation error and Lipschitz continuation advantage, the utility loss relative to the oracle stopper is bounded by the occupancy of an `\varepsilon`-tube around the threshold plus a small estimator-failure term.

This is elegant and directly connected to experiments.

---

# 11. Immediate next step

Write a short proposition deriving entropy error from posterior-score error, then create a notation sheet that unifies Theorem A and Theorem B with the algorithm section.
