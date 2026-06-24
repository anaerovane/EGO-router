# EGO Theory Scaffold v1

## Goal of this note
This document narrows the full EGO formulation into theorem-ready cores. The objective is not yet to prove the strongest possible results, but to identify the cleanest statements that are both mathematically defensible and relevant to the full algorithm.

We will proceed from a simplified scalar stopping model to richer multi-action orchestration.

---

# 1. Simplified one-dimensional stopping model

## 1.1 Latent confidence state
Let `z_t \in [0,1]` denote the controller's latent confidence that the current best answer is correct at round `t`.

Equivalent uncertainty can be measured as
\[
u_t = 1 - z_t.
\]

Higher `z_t` means the controller is more certain it already has the right answer.

## 1.2 Actions
At each step the controller chooses either:
- `CONTINUE`
- `STOP`

Stopping yields terminal reward
\[
G(z_t, b_t) = z_t - \kappa_{\text{stop}}(b_t),
\]
where `\kappa_{\text{stop}}(b_t)` can encode penalties for stopping under tight or loose budgets.

Continuing yields
\[
\mathbb{E}[V_{t+1}(z_{t+1}, b_{t+1}) \mid z_t, b_t] - c(b_t),
\]
where `c(b_t) > 0` is the immediate continuation cost.

The Bellman equation is
\[
V_t(z,b) = \max\Big\{G(z,b),\; -c(b) + \mathbb{E}[V_{t+1}(z', b') \mid z,b]\Big\}.
\]

---

# 2. Conditions for threshold stopping

The target result is that for fixed budget `b`, there exists a threshold `\theta(b)` such that it is optimal to stop iff `z_t \ge \theta(b)`.

## Assumption S1: monotone confidence improvement
Conditioned on continuing, the next confidence `z_{t+1}` first-order stochastically increases with `z_t`.

Interpretation: if we are already in a more informed state, additional evidence tends not to make us systematically less informed.

## Assumption S2: diminishing marginal value of continuation
Define the continuation advantage
\[
\Delta_t(z,b) = -c(b) + \mathbb{E}[V_{t+1}(z',b') \mid z,b] - G(z,b).
\]
Assume `\Delta_t(z,b)` is non-increasing in `z`.

Interpretation: once confidence is already high, another step brings less marginal benefit.

## Assumption S3: positive continuation cost
For every feasible state,
\[
c(b) \ge c_{\min} > 0.
\]

## Assumption S4: monotone terminal utility
`G(z,b)` is non-decreasing in `z`.

---

# 3. First main theorem candidate

## Theorem A (budget-dependent threshold stopping, simplified form)
Under Assumptions S1-S4, for each round `t` and feasible budget `b`, there exists a threshold `\theta_t(b) \in [0,1]` such that the optimal policy of the simplified stopping problem is:
\[
\pi_t^*(z,b) =
\begin{cases}
\texttt{STOP}, & z \ge \theta_t(b), \\
\texttt{CONTINUE}, & z < \theta_t(b).
\end{cases}
\]
Moreover, if continuation becomes more expensive as budget shrinks, then `\theta_t(b)` is non-increasing in available budget: with less budget remaining, the controller stops earlier.

### Proof sketch idea
- Show by backward induction that `V_t(z,b)` is increasing in `z`.
- Show the continuation advantage `\Delta_t(z,b)` is non-increasing in `z`.
- Therefore the set `{z : \Delta_t(z,b) \le 0}` is an interval, yielding a threshold rule.
- Budget monotonicity follows if `c(b)` increases as the budget shrinks or if the future value term decreases with shrinking budget.

This theorem is the cleanest entry point for the paper because it formalizes “stop when sufficiently confident, with confidence threshold depending on budget.”

---

# 4. From scalar confidence to entropy-based EGO

The full EGO controller does not maintain a single confidence scalar; it uses statistics
\[
s_t = (H_t, M_t, D_t, V_t, b_t).
\]

To bridge the gap, define an aggregate uncertainty score
\[
U_t = \alpha_H H_t - \alpha_M M_t + \alpha_D D_t - \alpha_V V_t.
\]

Lower `U_t` means greater confidence and lower need for further information. The threshold stopping claim then becomes:

## Theorem A' (aggregate-uncertainty stopping, aspirational form)
Suppose the future benefit of additional actions depends on the current state only through an aggregate uncertainty index `U_t`, and the continuation advantage is non-decreasing in `U_t`. Then there exists a budget-dependent threshold `\Theta(b_t)` such that stopping is optimal whenever `U_t \le \Theta(b_t)`.

This is the paper-facing theorem version more aligned with EGO.

---

# 5. Approximate stopping with imperfect posterior estimation

The real algorithm does not observe exact uncertainty; it estimates `H_t, M_t, D_t, V_t` from sampled candidates and verifier scores.

Let `\widehat U_t` denote the estimated uncertainty index and `U_t^*` the oracle index.
Assume a uniform approximation bound
\[
|\widehat U_t - U_t^*| \le \varepsilon
\]
with high probability.

## Theorem B (approximate stopping gap, candidate form)
Assume:
- the continuation advantage is `L`-Lipschitz in `U_t`;
- reward is bounded in `[0,1]`;
- horizon is at most `T`.

Then the utility gap between the estimated-threshold policy and the oracle-threshold policy satisfies
\[
\mathbb{E}[U_T(\pi^*) - U_T(\widehat \pi)] \le C_1 T \varepsilon + C_2 T \delta,
\]
where `\delta` is the failure probability of the uncertainty estimator and `C_1, C_2` are constants depending on Lipschitz and cost parameters.

### Interpretation
If the posterior and verifier are reasonably calibrated, EGO loses only linearly in the estimation error. This gives a mathematically clean reason to care about calibration.

---

# 6. Multi-action orchestration result

Now restore the richer action set:
\[
\mathcal{A} = \{\texttt{THINK}, \texttt{TOOL}(m), \texttt{DELEGATE}(k), \texttt{STOP}\}.
\]

Each non-stop action has estimated net value
\[
Q_t(a) = \operatorname{VOI}_t(a) - C_t(a).
\]

The policy is:
1. stop if uncertainty is below the stopping threshold;
2. otherwise choose the feasible action with maximal positive `Q_t(a)`;
3. stop if every feasible action has non-positive net value.

This suggests the following proposition.

## Proposition C (myopic optimality under adaptive submodularity, candidate form)
Suppose information gathered by actions is adaptively submodular and action costs are modular. Then the greedy EGO action-selection rule that chooses the feasible action with maximum marginal value-per-cost attains a constant-factor approximation to the optimal adaptive policy.

### Why this matters
This proposition would justify the action-routing component, not only the stopping component.

### Risk
This is mathematically attractive but requires careful task modeling. It may be better as an optional appendix result rather than the paper's first theorem.

---

# 7. Online-learning extension

If the value model is not known, one can estimate action values online.
Assume a realizable linear model:
\[
Q_t(a) = w_a^{*\top} \phi(s_t) + \xi_t,
\]
where `\xi_t` is conditionally sub-Gaussian.

## Theorem D (contextual-bandit regret, candidate form)
If EGO-v1 learns action parameters using a confidence-based linear contextual bandit update, then against the best stationary linear orchestration rule in hindsight, cumulative regret after `T` rounds satisfies
\[
\operatorname{Regret}(T) = \widetilde O\big(d \sqrt{T |\mathcal{A}|}\big),
\]
where `d` is the feature dimension.

This theorem is standard-ish, so it is not enough by itself for the paper, but it is useful as a clean extension.

---

# 8. Recommended theorem strategy for the paper

## Keep in the main paper
1. **Theorem A / A'**: threshold stopping exists and depends monotonically on budget.
2. **Theorem B**: approximate stopping suffers bounded utility loss under uncertainty-estimation error.

These two are well aligned with the paper's core message.

## Keep as optional / appendix / future work
3. **Proposition C**: adaptive-submodular justification of greedy action routing.
4. **Theorem D**: online-learning regret for learned action scoring.

This split avoids overloading the main paper.

---

# 9. Exact mathematical choices to freeze next

To move from scaffold to proof, we should now freeze:

1. **A scalar uncertainty proxy for the proof section**
   - easiest: `U_t = H_t`
   - richer but still manageable: `U_t = \alpha_H H_t - \alpha_M M_t`

2. **A threshold family**
   - simplest: stop if `H_t \le h_0 + \alpha / (B_t^{\text{step}} + 1)`
   - theory-friendly alternative: stop if `H_t \le h(B_t^{\text{step}})` for a monotone function `h`

3. **A minimal environment model**
   - each continue action reduces entropy by a random amount with decreasing expectation
   - stop reward is a monotone transform of remaining entropy

4. **A calibration model for Theorem B**
   - sup-norm error on entropy estimate
   - or bounded score perturbation inducing bounded entropy error

---

# 10. Immediate next outputs
- Write `03_algorithm_v1.md` with exact algorithmic choices that match Theorem A/B.
- Then write `04_experimental_blueprint.md` with synthetic environments explicitly constructed to satisfy or violate the assumptions.
