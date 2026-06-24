# EGO Formalization v1

## 1. Problem setting
We study a budgeted agent-orchestration problem. Given an input task `x`, an agentic system interacts with internal reasoning modules, external tools, and delegated expert agents over multiple rounds before returning a final answer `y`.

The main decision is not only *what answer to output*, but also *whether additional computation or interaction is worth its cost*.

We model this as a finite-horizon sequential decision problem with explicit stopping.

---

## 2. Interaction process
At round `t`, the controller has access to the trajectory

\[
\tau_t = (x, a_1, o_1, a_2, o_2, \dots, a_{t-1}, o_{t-1}),
\]

where each `a_i` is a controller action and each `o_i` is the corresponding observation.

Available actions are

\[
\mathcal{A} = \{\texttt{THINK}\} \cup \{\texttt{TOOL}(m): m \in \mathcal{M}\} \cup \{\texttt{DELEGATE}(k): k \in \mathcal{K}\} \cup \{\texttt{STOP}\}.
\]

Interpretation:
- `THINK`: perform an additional internal reasoning step or sample more candidate reasoning paths;
- `TOOL(m)`: invoke external tool `m`;
- `DELEGATE(k)`: query delegated expert `k`;
- `STOP`: terminate and emit the current best answer.

After taking any non-stop action `a_t`, the controller receives a random observation `o_t ~ P(· | \tau_t, a_t)` and updates the trajectory to `\tau_{t+1}`.

---

## 3. Budgeted state
A fully general history is too large to analyze directly. We therefore define a compact controller state

\[
s_t = (u_t, b_t),
\]

where:
- `u_t` is an uncertainty summary extracted from the current trajectory;
- `b_t` is the remaining budget.

### 3.1 Remaining budget
We use a vector-valued remaining budget

\[
b_t = (B_t^{\text{tok}}, B_t^{\text{lat}}, B_t^{\text{step}}),
\]

representing token budget, latency budget, and remaining decision rounds.

Each action `a` incurs expected cost vector

\[
c(a) = (c^{\text{tok}}(a), c^{\text{lat}}(a), c^{\text{step}}(a)).
\]

Feasible actions must satisfy `b_t - c(a) \succeq 0` coordinatewise.

### 3.2 Uncertainty summary
The uncertainty summary is

\[
u_t = (H_t, M_t, D_t, V_t).
\]

These components are:

1. **Predictive entropy**
\[
H_t = -\sum_{y \in \mathcal{Y}} p_t(y) \log p_t(y),
\]
where `p_t(y)` is the controller's current approximate posterior over final answers.

2. **Answer margin**
\[
M_t = p_t(y_{(1)}) - p_t(y_{(2)}),
\]
where `y_{(1)}` and `y_{(2)}` are the most and second-most likely answers.

3. **Inter-path disagreement**
`D_t` measures disagreement across multiple sampled reasoning paths, tool outputs, or delegated responses. In v1, it can be instantiated as one minus the empirical consensus rate or as a diversity score over candidate answers.

4. **Verifier confidence**
\[
V_t \in [0,1],
\]
which denotes the score assigned by a verifier / critic / answer-checking model to the current best candidate answer.

The compact state is therefore

\[
s_t = (H_t, M_t, D_t, V_t, B_t^{\text{tok}}, B_t^{\text{lat}}, B_t^{\text{step}}).
\]

---

## 4. Posterior approximation
Because the true posterior over answers is unavailable, EGO uses a sampled approximation.

At round `t`, the controller constructs a finite candidate set

\[
\mathcal{C}_t = \{\hat y_t^{(1)}, \dots, \hat y_t^{(K)}\}
\]

by sampling candidate answers from reasoning paths, tool-augmented completions, and delegated experts.

Each candidate receives a score `S_t(\hat y)` from a verifier or consensus mechanism. The approximate posterior is then

\[
p_t(y) = \frac{\exp(S_t(y))}{\sum_{y' \in \mathcal{C}_t} \exp(S_t(y'))}.
\]

This posterior induces `H_t`, `M_t`, and downstream stopping statistics.

---

## 5. Utility objective
The controller is rewarded for solving the task correctly while paying for resource consumption and risky actions.

Let `\widehat y_t` be the current best answer at time `t`. Upon stopping at time `T`, the system receives utility

\[
U_T = R(x, \widehat y_T) - \lambda_{\text{tok}} C_T^{\text{tok}} - \lambda_{\text{lat}} C_T^{\text{lat}} - \lambda_{\text{step}} C_T^{\text{step}} - \lambda_{\text{risk}} C_T^{\text{risk}},
\]

where:
- `R(x, \widehat y_T)` is task reward, e.g. 1 for correct and 0 for incorrect, or a more graded task-specific score;
- `C_T^{\text{tok}}, C_T^{\text{lat}}, C_T^{\text{step}}` are cumulative resource costs;
- `C_T^{\text{risk}}` penalizes unstable tool use, poor delegation, or unsafe behavior.

The controller's goal is

\[
\max_\pi \; \mathbb{E}_\pi[U_T].
\]

This is the core budget-aware orchestration objective.

---

## 6. Value of information view
To make decisions interpretable and theorem-friendly, EGO scores each non-stop action by estimated value of information minus cost.

For `a \neq \texttt{STOP}` define

\[
Q_t(a) = \operatorname{VOI}_t(a) - C_t(a),
\]

where the action cost scalar is

\[
C_t(a) = \lambda_{\text{tok}} c_t^{\text{tok}}(a) + \lambda_{\text{lat}} c_t^{\text{lat}}(a) + \lambda_{\text{step}} c_t^{\text{step}}(a) + \lambda_{\text{risk}} c_t^{\text{risk}}(a).
\]

The value-of-information term is approximated by expected improvement in uncertainty statistics:

\[
\operatorname{VOI}_t(a)
= \beta_H \mathbb{E}[H_t - H_{t+1} \mid s_t, a]
+ \beta_M \mathbb{E}[M_{t+1} - M_t \mid s_t, a]
+ \beta_V \mathbb{E}[V_{t+1} - V_t \mid s_t, a]
+ \beta_D \mathbb{E}[D_t - D_{t+1} \mid s_t, a].
\]

So an action is attractive when it is expected to reduce uncertainty, increase answer separation, improve verifier confidence, and reduce disagreement more than it costs.

---

## 7. Entropy-gated stopping
EGO stops for two complementary reasons:

1. uncertainty is already low enough relative to remaining budget;
2. no available action has positive net value.

### 7.1 Threshold gate
Define budget-dependent thresholds

\[
h(b_t), \quad d(b_t), \quad v(b_t).
\]

The entropy gate opens if further work is still warranted:

\[
g_t = \mathbf{1}\{H_t > h(b_t) \; \lor \; D_t > d(b_t) \; \lor \; V_t < v(b_t)\}.
\]

If `g_t = 0`, the controller stops.

### 7.2 Value-based stop
Even when the gate is open, the controller stops if no action has positive value:

\[
\max_{a \in \mathcal{A} \setminus \{\texttt{STOP}\}} Q_t(a) \le 0.
\]

Therefore the full stopping rule is

\[
\texttt{STOP if } g_t = 0 \; \text{or} \; \max_{a \ne \texttt{STOP}} Q_t(a) \le 0.
\]

---

## 8. EGO-v1 action scoring
For implementation, EGO-v1 uses a feature map

\[
\phi(s_t) = [H_t, M_t, D_t, V_t, B_t^{\text{tok}}, B_t^{\text{lat}}, B_t^{\text{step}}],
\]

and action-specific linear scores

\[
Q_t(a) = w_a^\top \phi(s_t) - C_t(a).
\]

This linear version is simple, interpretable, and compatible with contextual-bandit analysis.

Possible v1 actions:
- `THINK`
- `TOOL(search)`
- `DELEGATE(math)`
- `DELEGATE(code)`
- `STOP`

---

## 9. Theorem-friendly assumptions
To prepare theory, the following assumptions are natural.

### Assumption A1: monotone information gain
For any feasible non-stop action `a`,
\[
\mathbb{E}[H_{t+1} \mid s_t, a] \le H_t,
\]
and similarly disagreement does not increase in expectation too aggressively.

### Assumption A2: diminishing returns
For each action type, the marginal uncertainty reduction decreases as more information of that type has already been acquired.

### Assumption A3: strictly positive action cost
There exists `c_{\min} > 0` such that every feasible non-stop action satisfies
\[
C_t(a) \ge c_{\min}.
\]

### Assumption A4: calibrated posterior error is bounded
The sampled posterior approximation `p_t` differs from an ideal latent posterior by bounded error in a suitable metric.

These assumptions are enough to motivate threshold-style optimal stopping and approximate suboptimality bounds.

---

## 10. Candidate theorem directions
1. **Threshold existence theorem**: under monotone information gain and diminishing returns, an optimal stopping policy exists that is monotone in uncertainty and remaining budget.
2. **Approximate stopping bound**: if posterior approximation error is bounded by `\epsilon`, then the utility gap between EGO and an oracle stopper is bounded by a function of `\epsilon` and the horizon.
3. **Bandit-style regret bound**: when `Q_t(a)` follows a realizable linear model, online learning of `w_a` yields sublinear regret against the best stationary orchestration rule in hindsight.

---

## 11. Immediate next formal tasks
- Turn the above into a notation table.
- Choose one exact definition of disagreement `D_t` for theory.
- Choose one exact family of threshold functions `h(b), d(b), v(b)`.
- State Theorem A precisely in a one-dimensional simplified setting first.
- Build the synthetic environment to match the assumptions.
