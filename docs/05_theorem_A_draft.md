# Theorem A Draft: Budget-Dependent Threshold Stopping

## Objective
State the first main theorem in a form that is simple enough to prove cleanly, but still faithful to the EGO story.

The theorem should justify the claim:
> if further reasoning/tooling/delegation has diminishing marginal value and positive cost, then there exists a budget-dependent stopping threshold.

---

# 1. Simplified proof model

We consider a finite-horizon dynamic program with state `(u, b)` where:
- `u \in [0, \bar u]` is scalar uncertainty; lower is better;
- `b \in \{0,1,2,\dots,B\}` is remaining step budget.

At each state the controller chooses one of two abstract actions:
- `STOP`
- `CONTINUE`

This theorem intentionally abstracts away which non-stop action is chosen. It only proves when it is optimal to stop.

## 1.1 Terminal reward for stopping
Stopping at state `(u,b)` yields
\[
G(u,b),
\]
where `G` is non-increasing in uncertainty `u`.

Interpretation: the less uncertain we are, the higher the expected answer quality at stop time.

## 1.2 Continuation value
Continuing from `(u,b)` with `b \ge 1` yields
\[
-c(b) + \mathbb{E}[V_{b-1}(u') \mid u, b],
\]
where:
- `c(b) > 0` is immediate continuation cost,
- `u'` is next-step uncertainty after gathering one more piece of information,
- `V_b(u)` is the optimal value function with uncertainty `u` and remaining budget `b`.

The Bellman equation is
\[
V_b(u) = \max\left\{G(u,b),\; -c(b) + \mathbb{E}[V_{b-1}(u') \mid u,b]\right\}.
\]

Boundary case: `V_0(u) = G(u,0)`.

---

# 2. Assumptions

## Assumption T1: monotone stopping payoff
For each budget `b`, the stopping payoff `G(u,b)` is non-increasing in `u`.

## Assumption T2: uncertainty improves in expectation
For each feasible budget `b >= 1`, the conditional law of `u'` given `u` is stochastically non-increasing in `u`.

Interpretation: if you start from a lower-uncertainty state, then after one more information-gathering step you remain stochastically better.

## Assumption T3: diminishing continuation advantage
Define the continuation advantage
\[
\Delta_b(u) := -c(b) + \mathbb{E}[V_{b-1}(u') \mid u,b] - G(u,b).
\]
Assume `\Delta_b(u)` is non-increasing in `u`.

Interpretation: the more uncertain we already are? Wait, since lower `u` is better, this means as uncertainty rises, continuation becomes weakly more attractive. Equivalently, as certainty increases, the marginal value of additional work falls.

## Assumption T4: positive cost
There exists `c_{\min} > 0` such that `c(b) \ge c_{\min}` for all `b >= 1`.

## Assumption T5: budget monotonicity of future value
For each fixed `u`, `V_b(u)` is non-decreasing in `b`.

Interpretation: having more steps remaining never hurts.

---

# 3. Main theorem statement

## Theorem A
Under Assumptions T1-T5, for every remaining budget `b`, there exists a threshold `\theta_b \in [0, \bar u] \cup \{\infty\}` such that an optimal policy is:
\[
\pi_b^*(u) =
\begin{cases}
\texttt{STOP}, & u \le \theta_b,\\
\texttt{CONTINUE}, & u > \theta_b.
\end{cases}
\]
That is, there exists an optimal uncertainty-threshold stopping rule.

Moreover, if the continuation cost becomes effectively larger as budget shrinks, then the thresholds are monotone in budget:
\[
\theta_{b+1} \le \theta_b,
\]
meaning with more remaining budget, the controller is willing to continue down to lower uncertainty; with less budget, it stops earlier.

### Mapping back to EGO
- uncertainty `u` corresponds to entropy or an aggregate uncertainty score;
- the theorem explains why budget-aware entropy thresholding is a principled design, not a heuristic hack.

---

# 4. Proof sketch

## Step 1: show value monotonicity
Prove by induction on `b` that `V_b(u)` is non-increasing in `u`.

- Base case `b=0` follows from Assumption T1 because `V_0(u)=G(u,0)`.
- Inductive step uses stochastic monotonicity of `u'` and monotonicity of `V_{b-1}`.

## Step 2: analyze continuation advantage
By definition,
\[
\Delta_b(u) = -c(b) + \mathbb{E}[V_{b-1}(u')\mid u,b] - G(u,b).
\]
Under Assumption T3, this is non-increasing in `u`.

Therefore the set
\[
\mathcal{S}_b := \{u : \Delta_b(u) \le 0\}
\]
is an interval of the form `[0, \theta_b]`.

## Step 3: characterize optimal action
The Bellman equation chooses `STOP` exactly when `\Delta_b(u) \le 0`, so the optimal stopping region is `[0, \theta_b]`.

## Step 4: budget monotonicity
Use Assumption T5 to show that with more budget the continuation term weakly increases. Therefore continuation remains attractive over a larger set of uncertainty states, implying threshold monotonicity.

---

# 5. Discussion of modeling choice

## Why uncertainty is defined so that lower is better
This makes the stopping threshold align naturally with entropy-based rules: stop when uncertainty is low enough.

## Why we first abstract away multiple action types
The first theorem should isolate the stopping logic. If we try to prove action routing and stopping simultaneously, the presentation will become much harder.

## Where multi-action routing enters later
The theorem only says *whether* to continue. Once continuation is justified, EGO chooses *how* to continue by comparing action values.

---

# 6. Corollary for entropy-based EGO

Let `u = H`, predictive entropy. Suppose the one-step effect of the chosen continuation action yields next entropy `H'` such that:
- `H'` is stochastically non-increasing in `H`,
- the continuation advantage is non-increasing in `H`.

Then an optimal policy exists that stops whenever
\[
H_t \le h^*(B_t^{step})
\]
for some budget-dependent threshold function `h^*`.

This is the exact formal justification for EGO's entropy gate.

---

# 7. What still needs tightening before proof-writing

1. We should make Assumption T3 derivable from more primitive assumptions instead of postulating it directly.
2. We should decide whether theorem notation uses uncertainty `u` or confidence `z`; uncertainty is cleaner for entropy alignment.
3. We should determine whether budget monotonicity should be strict or weak.
4. We should construct a synthetic environment where all assumptions are transparently satisfied.

---

# 8. Recommended next theorem step

Write a companion proposition that gives primitive sufficient conditions for Assumption T3. For example:
- `G(u,b)` smooth and concave in `-u`,
- expected uncertainty reduction from one extra step decreases as uncertainty falls,
- continuation cost is action-independent in the simplified theorem.

That proposition will make the theorem feel less assumption-heavy and more publishable.
