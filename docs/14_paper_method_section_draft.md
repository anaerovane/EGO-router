# Paper Draft: Method Section

## 3 Method

We propose **Entropy-Gated Orchestration (EGO)**, a budget-aware controller for LLM-based agents. Instead of hard-coding a fixed workflow such as “reason, then tool, then answer,” EGO treats orchestration itself as a sequential decision problem: at each round, the controller decides whether to continue gathering information, which information-gathering action to take, and when to stop and emit a final answer.

The key design goal is to unify four decisions under one control framework:

1. whether to continue or stop,
2. whether additional internal reasoning is worthwhile,
3. whether an external tool call is worthwhile, and
4. whether delegating to a specialized expert is worthwhile.

This section first formalizes the orchestration problem, then introduces EGO's uncertainty-aware stopping rule and unified routing score, and finally presents a learned routing extension based on contextual bandits.

---

### 3.1 Problem formulation

Given an input task $x$, an agentic system interacts with internal reasoning modules, external tools, and delegated expert agents over multiple rounds before producing a final answer $\hat y$. Let the controller action space be

\[
\mathcal{A} = \{\texttt{THINK}\} \cup \{\texttt{TOOL}(m): m \in \mathcal{M}\} \cup \{\texttt{DELEGATE}(k): k \in \mathcal{K}\} \cup \{\texttt{STOP}\}.
\]

Here:

- $\texttt{THINK}$ performs one additional internal refinement step;
- $\texttt{TOOL}(m)$ invokes tool $m$ to gather external evidence;
- $\texttt{DELEGATE}(k)$ queries expert $k$ for a specialized candidate answer;
- $\texttt{STOP}$ terminates the process and returns the current best candidate.

At round $t$, the controller has access to a trajectory

\[
\tau_t = (x, a_1, o_1, \dots, a_{t-1}, o_{t-1}),
\]

where $a_i$ is the chosen action and $o_i$ is the resulting observation. Since the full trajectory is too large for practical control, EGO summarizes it into a compact uncertainty-and-budget state.

---

### 3.2 Uncertainty-aware state representation

EGO uses the following compact state:

\[
s_t = (H_t, M_t, D_t, V_t, B_t^{\mathrm{tok}}, B_t^{\mathrm{lat}}, B_t^{\mathrm{step}}).
\]

The uncertainty components are:

- **Predictive entropy** $H_t$, measuring uncertainty in the current answer posterior;
- **Answer margin** $M_t$, the probability gap between the top-1 and top-2 candidates;
- **Disagreement** $D_t$, measuring inconsistency across reasoning paths, tool-supported candidates, or delegated expert outputs;
- **Verifier confidence** $V_t$, the verifier score assigned to the current best candidate.

The budget components are:

- remaining token budget $B_t^{\mathrm{tok}}$,
- remaining latency budget $B_t^{\mathrm{lat}}$,
- remaining step budget $B_t^{\mathrm{step}}$.

This representation is deliberately simple: it is expressive enough to support principled stopping and routing, but compact enough to admit theorem-friendly analysis and lightweight implementation.

---

### 3.3 Posterior estimation from candidate answers

At each round, EGO maintains a candidate pool

\[
\mathcal{C}_t = \{\hat y_t^{(1)}, \dots, \hat y_t^{(K_t)}\},
\]

whose elements are generated from different channels: the initial draft, internal refinement, tool-augmented reasoning, and delegated experts. Each candidate $y \in \mathcal{C}_t$ receives a score $S_t(y)$, which may incorporate verifier assessment, support from multiple candidate-generation channels, and optionally freshness of external evidence.

EGO then constructs an approximate posterior over candidate answers by a temperature-scaled softmax:

\[
p_t(y) = \frac{\exp(S_t(y)/\tau)}{\sum_{y' \in \mathcal{C}_t} \exp(S_t(y')/\tau)}.
\]

This posterior induces the uncertainty statistics used by the controller. In particular,

\[
H_t = -\sum_{y \in \mathcal{C}_t} p_t(y) \log p_t(y),
\]

\[
M_t = p_t(y_{(1)}) - p_t(y_{(2)}),
\]

where $y_{(1)}$ and $y_{(2)}$ are the highest- and second-highest-probability candidates. In our implementation, disagreement is estimated from empirical consensus among candidates, and verifier confidence is obtained by mapping the verifier score of the best candidate into $[0,1]$.

This candidate-based posterior view is important because it connects the controller to practical agent systems: EGO does not assume access to a true posterior, only to a pool of candidate answers and a way to score them.

---

### 3.4 Utility objective

EGO optimizes a utility that trades off final answer quality against resource consumption. If the controller stops at round $T$ and returns candidate $\hat y_T$, the utility is

\[
U_T = R(x, \hat y_T)
- \lambda_{\mathrm{tok}} C_T^{\mathrm{tok}}
- \lambda_{\mathrm{lat}} C_T^{\mathrm{lat}}
- \lambda_{\mathrm{step}} C_T^{\mathrm{step}}
- \lambda_{\mathrm{risk}} C_T^{\mathrm{risk}},
\]

where $R(x, \hat y_T)$ is a task reward and the $C_T$ terms denote cumulative token, latency, step, and risk costs. The controller seeks a policy $\pi$ maximizing the expected utility:

\[
\max_\pi \; \mathbb{E}_\pi[U_T].
\]

This objective highlights the central orchestration tradeoff: additional actions are useful only if the expected information gain they provide outweighs their cost.

---

### 3.5 Budget-aware stopping

A core contribution of EGO is that stopping is treated as a first-class control problem rather than a fixed heuristic. EGO stops for two complementary reasons: either uncertainty is already low enough given the remaining budget, or no continuation action has positive net value.

#### 3.5.1 Uncertainty gate

Let $h(\cdot)$ denote a budget-dependent entropy threshold. In the simplest version, we parameterize it as a function of the remaining step budget:

\[
h(B_t^{\mathrm{step}}) = h_0 + \frac{\alpha_h}{B_t^{\mathrm{step}} + 1}.
\]

This form captures an intuitive behavior: when the remaining budget is small, the controller should be willing to stop at higher uncertainty; when the remaining budget is larger, the controller can afford to continue until uncertainty falls further. Equivalently, the stopping threshold decreases as more budget remains.

In the practical controller, we combine entropy with additional uncertainty guards based on disagreement and verifier confidence. Define the gate indicator

\[
g_t = \mathbf{1}\{H_t > h(B_t^{\mathrm{step}}) \lor D_t > d(B_t^{\mathrm{step}}) \lor V_t < v(B_t^{\mathrm{step}})\},
\]

where $d(\cdot)$ and $v(\cdot)$ are optional disagreement and verifier thresholds. If $g_t = 0$, the controller stops.

#### 3.5.2 Value-based stopping

Even if the uncertainty gate remains open, continuing is not always worthwhile. Let $Q_t(a)$ denote the net value of continuation action $a \neq \texttt{STOP}$. If

\[
\max_{a \in \mathcal{A} \setminus \{\texttt{STOP}\}} Q_t(a) \le 0,
\]

then the controller stops because no feasible continuation action offers positive utility.

Combining the two criteria, EGO uses the stopping rule

\[
\texttt{STOP if } g_t = 0 \quad \text{or} \quad \max_{a \neq \texttt{STOP}} Q_t(a) \le 0.
\]

This two-level formulation is important: the threshold gate provides a theorem-friendly stopping structure, while the value-based stop prevents wasteful continuation even when uncertainty remains nontrivial.

---

### 3.6 Unified action scoring for think, tool, and delegate

When continuation is still worthwhile, EGO chooses among internal reasoning, tool use, and expert delegation using a unified net-value score. For any continuation action $a$, we define

\[
Q_t(a) = \widehat{\mathrm{VOI}}_t(a) - C_t(a),
\]

where $\widehat{\mathrm{VOI}}_t(a)$ is an estimated value of information and $C_t(a)$ is the action cost.

The cost term can combine token, latency, step, and risk penalties:

\[
C_t(a) = \lambda_{\mathrm{tok}} c_t^{\mathrm{tok}}(a)
+ \lambda_{\mathrm{lat}} c_t^{\mathrm{lat}}(a)
+ \lambda_{\mathrm{step}} c_t^{\mathrm{step}}(a)
+ \lambda_{\mathrm{risk}} c_t^{\mathrm{risk}}(a).
\]

In the heuristic version of EGO, the value-of-information term is approximated by a lightweight gain proxy derived from the current state. Intuitively:

- high entropy increases the value of all continuation actions;
- high disagreement increases the value of tool and expert actions that can resolve uncertainty;
- low verifier confidence increases the value of additional information gathering;
- action relevance and prior relevance help distinguish when a particular tool or expert is likely to be useful.

The controller then chooses

\[
a_t = \arg\max_{a \in \mathcal{A} \setminus \{\texttt{STOP}\}} Q_t(a),
\]

subject to feasibility under the remaining budget.

This unified scoring is a key design choice: it places internal reasoning, tool invocation, and delegated expertise into a common decision space, rather than treating them as unrelated modules.

---

### 3.7 EGO-v1 heuristic instantiation

Our first implementation, EGO-v1, uses a deliberately simple and interpretable action scorer. Given state features

\[
\phi_t = [1, H_t, M_t, D_t, V_t, B_t^{\mathrm{step}}],
\]

the controller estimates action value with a hand-crafted linearized proxy of the form

\[
Q_t(a) \approx w_a^\top \phi_t - C_t(a).
\]

We initialize the action preferences according to the following qualitative rules:

- `THINK` is attractive when entropy is moderate and continuation is cheap;
- `TOOL(search)` is attractive when uncertainty is high and external evidence is likely needed;
- `DELEGATE(math)` is attractive for theorem-like or derivation-heavy tasks;
- `DELEGATE(code)` is attractive for implementation and debugging tasks.

The purpose of EGO-v1 is not to deliver the strongest possible routing policy, but to provide a clean, analyzable baseline that already covers the full action space of interest.

---

### 3.8 Learned routing extension

While the heuristic scorer is interpretable, it does not adapt online. To support learnable specialization, we extend EGO with a contextual-bandit-style routing module.

For each action $a$, we build a feature vector including uncertainty statistics, remaining budget, action cost, relevance, prior relevance, action-type indicators, and a small set of interaction terms. The learned routing score is

\[
\mathrm{score}_t(a)
=
\hat r_\theta(s_t, a)
+ \mathrm{bonus}_t(s_t, a)
+ \lambda \cdot \widehat{\mathrm{gain}}_t(a)
- C_t(a),
\]

where:

- $\hat r_\theta(s_t, a)$ is a learned reward predictor,
- $\mathrm{bonus}_t(s_t, a)$ is an exploration bonus,
- $\widehat{\mathrm{gain}}_t(a)$ is the heuristic gain proxy,
- $C_t(a)$ is the action cost.

In our current prototype, we instantiate this with a per-action LinUCB model. Each action maintains its own ridge-regularized linear parameters, and the controller updates them online using a proxy reward signal derived from verifier-confidence improvement and entropy reduction after executing the chosen action.

We deliberately adopt a **hybrid score** rather than a purely learned one. In early-stage agent settings, purely learned routing can be unstable due to sparse or noisy reward signals. Mixing learned predictions with a heuristic prior yields a more stable controller while still supporting a clear online-learning narrative.

---

### 3.9 Practical instantiation in agent stacks

Although EGO is defined abstractly, it is designed for direct use in real agent pipelines. Our implementation separates the method into three reusable components:

1. **Stopping core.** `EGOStoppingController` receives uncertainty metrics and remaining budget, and returns a stop/continue decision together with the stopping reason.
2. **Posterior estimator.** `CandidatePosteriorEstimator` converts a candidate pool and verifier scores into entropy, margin, disagreement, and verifier confidence.
3. **Framework adapter.** `LangChainEGOAgent` wraps invoke-style LLMs, tools, and expert agents, applies EGO control at each step, and logs the resulting trajectory.

This decomposition keeps the method framework-agnostic: EGO can be integrated into LangChain-like systems without coupling the research logic to a specific orchestration library.

---

### 3.10 Discussion and design rationale

EGO is designed around three principles.

**First, stopping should be principled.** Many agent systems implicitly assume a fixed number of reasoning or tool-use steps. EGO instead makes stopping depend on both uncertainty and remaining budget.

**Second, action families should be unified.** Internal reasoning, tool use, and delegation are often implemented as separate heuristics. EGO evaluates them under a common net-value formulation.

**Third, routing should be learnable.** Hand-crafted heuristics are useful for bootstrapping, but mixed-task environments require specialization. The learned routing extension gives EGO a path from interpretable heuristics to adaptive orchestration.

Together, these components turn orchestration itself into the object of optimization.

---

### 3.11 Summary

In summary, EGO formulates agent orchestration as a budget-aware sequential decision problem with explicit stopping and unified routing. Its core ingredients are:

- a compact uncertainty-and-budget state,
- candidate-based posterior estimation,
- budget-aware threshold stopping,
- unified action scoring over think/tool/delegate,
- and a learned routing extension for online adaptation.

This method design directly supports the theory and experiments in the following sections: threshold-style stopping aligns with our stopping analysis, while the learned scorer supports controlled routing experiments and future regret-style extensions.
