# EGO-v1 Algorithm Spec

## Purpose
This note freezes the first implementable and theorem-aligned version of EGO. The priority is not maximal performance, but a clean algorithm that is:
- mathematically interpretable,
- easy to simulate,
- compatible with threshold stopping theory,
- easy to ablate.

---

# 1. Scope of EGO-v1

We focus on single-instance final-answer tasks. A task terminates when the controller outputs one final answer.

EGO-v1 uses:
- one controller;
- a small fixed action set;
- sampled candidate answers to estimate uncertainty;
- a verifier to score candidate answers;
- explicit token / latency / step costs.

This deliberately avoids complex memory mechanisms and long tool chains in v1.

---

# 2. Action set

EGO-v1 action set:
\[
\mathcal{A}_{v1} = \{\texttt{THINK},\; \texttt{TOOL(search)},\; \texttt{DELEGATE(math)},\; \texttt{DELEGATE(code)},\; \texttt{STOP}\}.
\]

Interpretation:
- `THINK`: sample one additional internal reasoning path;
- `TOOL(search)`: call a retrieval/search tool to gather external evidence;
- `DELEGATE(math)`: ask a math-focused expert policy for a candidate answer and rationale;
- `DELEGATE(code)`: ask a code/tool-focused expert policy for a candidate answer and rationale;
- `STOP`: emit the current best answer.

Why this action set:
- it is small enough for clean experiments;
- it captures the three core non-stop behaviors: reasoning, tooling, delegation;
- it already supports heterogeneous tasks where routing matters.

---

# 3. State representation

We freeze the controller state as
\[
s_t = (H_t, M_t, D_t, V_t, B_t^{\text{tok}}, B_t^{\text{lat}}, B_t^{\text{step}}).
\]

## 3.1 Exact choices in v1

### Entropy
\[
H_t = -\sum_{y \in \mathcal{C}_t} p_t(y) \log p_t(y).
\]

### Margin
\[
M_t = p_t(y_{(1)}) - p_t(y_{(2)}).
\]

### Disagreement
For v1 we define
\[
D_t = 1 - \frac{\max_y n_t(y)}{K_t},
\]
where:
- `K_t` is the number of candidate answers collected so far,
- `n_t(y)` is the number of times candidate answer `y` appears.

This is simply one minus the empirical consensus rate.

### Verifier confidence
\[
V_t = \sigma(S_t(y_{(1)})),
\]
where `S_t(y_{(1)})` is the verifier score of the current best answer and `\sigma` maps it to `[0,1]`.

This choice is intentionally simple and robust.

---

# 4. Posterior construction

At each round, EGO-v1 maintains a candidate pool `\mathcal{C}_t`.
Each action contributes zero or more new candidates.

## 4.1 Candidate generation
- `THINK`: generate one new internal candidate answer.
- `TOOL(search)`: retrieve evidence, then generate one updated answer conditioned on the evidence.
- `DELEGATE(math)`: obtain one candidate answer from the math expert.
- `DELEGATE(code)`: obtain one candidate answer from the code expert.

## 4.2 Candidate scoring
Each candidate `y` receives a score
\[
S_t(y) = \gamma_1 \cdot \text{verifier}(y) + \gamma_2 \cdot \text{support}(y) + \gamma_3 \cdot \text{recency}(y),
\]
where:
- `verifier(y)` checks whether the candidate is well-supported / internally consistent;
- `support(y)` counts how many candidate-generation channels agree with `y`;
- `recency(y)` can mildly favor candidates updated with fresh evidence.

## 4.3 Approximate posterior
\[
p_t(y) = \frac{\exp(S_t(y)/\tau)}{\sum_{y' \in \mathcal{C}_t} \exp(S_t(y')/\tau)}.
\]

Hyperparameters:
- `\tau > 0`: temperature.
- For first experiments, set `\gamma_1 = 1`, `\gamma_2 = 0.5`, `\gamma_3 = 0`.

---

# 5. Cost model

Each non-stop action has deterministic nominal cost in v1.

\[
C_t(a) = \lambda_{\text{tok}} c^{\text{tok}}(a)
+ \lambda_{\text{lat}} c^{\text{lat}}(a)
+ \lambda_{\text{step}} c^{\text{step}}(a)
+ \lambda_{\text{risk}} c^{\text{risk}}(a).
\]

## Suggested nominal costs
- `THINK`: low token, low latency, one step
- `TOOL(search)`: medium token, high latency, one step, medium risk
- `DELEGATE(math)`: medium token, medium latency, one step, low-to-medium risk
- `DELEGATE(code)`: medium token, medium latency, one step, low-to-medium risk

For synthetic experiments we should set these explicitly.

---

# 6. Uncertainty gate

We freeze the budget-aware threshold family to depend only on remaining steps in v1. This keeps theory simple.

## 6.1 Thresholds
\[
h(B_t^{\text{step}}) = h_0 + \frac{\alpha_h}{B_t^{\text{step}} + 1},
\]
\[
d(B_t^{\text{step}}) = d_0 + \frac{\alpha_d}{B_t^{\text{step}} + 1},
\]
\[
v(B_t^{\text{step}}) = v_0 - \frac{\alpha_v}{B_t^{\text{step}} + 1}.
\]

Interpretation:
- with fewer steps remaining, EGO accepts higher entropy and higher disagreement before deciding to stop;
- equivalently, it becomes easier to stop when budget is almost exhausted.

## 6.2 Gate rule
Define
\[
g_t = \mathbf{1}\{H_t > h(B_t^{\text{step}}) \lor D_t > d(B_t^{\text{step}}) \lor V_t < v(B_t^{\text{step}})\}.
\]

If `g_t = 0`, then EGO stops immediately.

This makes entropy the primary proof variable, while disagreement and verifier confidence act as practical guards.

---

# 7. Action scoring rule

EGO-v1 uses a linearized approximation to value of information:
\[
Q_t(a) = w_a^\top \phi_t - C_t(a),
\]
where
\[
\phi_t = [1, H_t, M_t, D_t, V_t, B_t^{\text{step}}].
\]

We intentionally leave out token and latency budgets from the first scoring feature map because step budget is easiest to analyze. They can still appear in the cost term.

## 7.1 Sign expectations for weights
For sensible initialization:
- weight on `H_t` should be positive for `THINK`, `TOOL`, `DELEGATE`;
- weight on `M_t` should be negative;
- weight on `D_t` should be positive;
- weight on `V_t` should be negative;
- weight on remaining steps may be positive because extra budget increases option value.

## 7.2 Initialization strategy
Before learning, initialize weights by hand:
- `THINK`: best when entropy is moderate and cost is cheap;
- `TOOL(search)`: best when entropy/disagreement are high and external evidence is likely needed;
- `DELEGATE(math)`: best when the task is math-like;
- `DELEGATE(code)`: best when the task is programmatic / tool-mediated.

In synthetic experiments, task type indicators can be added if needed, but for the core theorem story we avoid this first.

---

# 8. Final decision rule

At round `t`:

1. Estimate `H_t, M_t, D_t, V_t`.
2. If the uncertainty gate is closed, return `STOP`.
3. For each feasible non-stop action, compute `Q_t(a)`.
4. If `max_a Q_t(a) <= 0`, return `STOP`.
5. Otherwise execute
\[
a_t = \arg\max_{a \in \mathcal{A}_{v1} \setminus \{\texttt{STOP}\}} Q_t(a).
\]
6. Update candidate pool, posterior, and budgets.

This preserves the two-level semantics:
- threshold-style stopping,
- value-based routing among continuation actions.

---

# 9. Pseudocode

```text
Algorithm EGO-v1
Input: task x, initial budget b0, candidate generator, verifier
Initialize candidate pool C0 from one draft answer
for t = 0,1,2,... while B_step > 0:
    score all candidates in Ct
    compute posterior pt(y)
    compute Ht, Mt, Dt, Vt

    if Ht <= h(B_step) and Dt <= d(B_step) and Vt >= v(B_step):
        return best candidate

    for each feasible action a in {THINK, TOOL(search), DELEGATE(math), DELEGATE(code)}:
        compute Qt(a) = wa^T phi_t - C(a)

    if max_a Qt(a) <= 0:
        return best candidate

    choose a_t = argmax_a Qt(a)
    execute a_t and add resulting candidate(s) to Ct+1
    update budget

return best candidate
```

---

# 10. Variants to compare later

To support ablation and clean empirical claims, we freeze the following variants:

## Variant 1: Threshold-only EGO
Use the entropy/disagreement/verifier gate, but among non-stop actions always choose a fixed heuristic priority.

## Variant 2: VOI-only EGO
Ignore the threshold gate and stop only when all action values are non-positive.

## Variant 3: Full EGO-v1
Use both the gate and value-based routing.

## Variant 4: Entropy-only EGO
Use only `H_t` in the state and stopping rule.

These variants are useful because they isolate exactly what the paper claims is novel.

---

# 11. What is frozen for theorem alignment

We now freeze the proof-facing core of EGO-v1 as follows:

1. primary scalar proof variable: `H_t`;
2. budget variable in theory: `B_t^{step}`;
3. stopping threshold family: `h(B_t^{step}) = h_0 + alpha_h / (B_t^{step} + 1)`;
4. approximate stopping condition in full implementation may also use `D_t` and `V_t`, but the first theorem can be proved using entropy only;
5. continuation action scores are linear in state features minus cost.

This means the first proof section can be built around entropy thresholding, while the algorithm section still presents the richer practical controller.

---

# 12. Next implementation-facing tasks
- Build the synthetic simulator around hidden task types and action-specific information gain.
- Decide how each action changes latent correctness probability.
- Decide how verifier noise enters posterior estimation.
- Map simulator outputs to candidate answers so `H_t`, `M_t`, `D_t`, `V_t` are measurable.
