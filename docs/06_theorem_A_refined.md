# Theorem A Refined: Entropy-Threshold Stopping Under Budget Constraints

## Purpose
This note refines the first main theorem into a proof-ready statement for the paper. The main goals are:
- replace assumption-heavy language with more primitive sufficient conditions;
- align the theorem directly with entropy-threshold stopping in EGO;
- isolate a clean proof strategy that can appear in the main text.

The final paper should present a simple theorem in the main body and move technical details to the appendix. This document is the bridge.

---

# 1. Proof-facing abstraction

We consider a finite-horizon stopping problem indexed by remaining step budget `b \in \{0,1,\dots,B\}`.
The controller state is a scalar uncertainty level
\[
H \in [0, \bar H],
\]
which should be interpreted as predictive entropy.

Lower entropy means greater confidence in the current best answer.

At state `(H,b)`, the controller chooses either:
- `STOP`, or
- `CONTINUE`.

The theorem only concerns the stopping decision. The choice of *which* continuation action to take will be analyzed separately.

---

# 2. Dynamic program

## 2.1 Stopping payoff
If the controller stops at entropy level `H` with remaining budget `b`, it receives payoff
\[
G(H,b).
\]
We interpret `G(H,b)` as the expected answer quality minus any terminal penalty. In the simplest case, `G(H,b)` depends on `H` only through the expected correctness of the current best candidate.

## 2.2 Continuation payoff
If the controller continues and `b \ge 1`, it pays an immediate cost `c(b) > 0` and transitions to a new entropy level `H'` according to a kernel
\[
K_b(\cdot \mid H).
\]
The continuation value is
\[
-c(b) + \mathbb{E}[V_{b-1}(H') \mid H].
\]

## 2.3 Bellman equation
The optimal value function satisfies
\[
V_b(H)
=
\max\left\{ G(H,b),\; -c(b) + \int V_{b-1}(H') K_b(dH' \mid H) \right\},
\]
with boundary condition
\[
V_0(H) = G(H,0).
\]

---

# 3. Primitive sufficient conditions

We now state assumptions that are more primitive than directly assuming monotonicity of the continuation advantage.

## Assumption R1: stopping quality improves as entropy decreases
For each fixed budget `b`, the stopping payoff `G(H,b)` is non-increasing in `H`.

Interpretation: if your posterior is sharper, your current answer is expected to be better.

## Assumption R2: one-step information is monotone in current entropy
For each `b \ge 1`, the transition kernel `K_b(\cdot \mid H)` is stochastically monotone in `H` in the following sense:
for any bounded non-increasing function `f`,
\[
\int f(H') K_b(dH' \mid H_1) \,\ge\, \int f(H') K_b(dH' \mid H_2)
\qquad \text{whenever } H_1 \le H_2.
\]

Interpretation: starting from lower entropy cannot make the distribution of next-step entropy worse in first-order stochastic order.

## Assumption R3: diminishing expected entropy reduction
Define the expected post-action entropy
\[
m_b(H) := \mathbb{E}[H' \mid H].
\]
Assume `m_b(H)` is non-decreasing and convex in `H`.

Equivalent interpretation:
- more uncertain states remain more uncertain after one extra step;
- marginal entropy reduction from continuation decreases as entropy becomes smaller.

A canonical example is
\[
H' = H - \rho_b(H) + \xi,
\]
where `\rho_b(H)` is non-negative, non-decreasing, and concave in `H`, and `\xi` is centered bounded noise.

## Assumption R4: positive continuation cost
For every `b \ge 1`,
\[
c(b) \ge c_{\min} > 0.
\]

## Assumption R5: option value of larger budget
For every fixed entropy `H`,
\[
V_b(H) \le V_{b+1}(H).
\]
This is natural since one may always mimic a shorter-budget policy.

## Assumption R6: regularity of stopping payoff
For each `b`, `G(H,b)` is continuous in `H`.
This ensures the stopping region can be expressed cleanly via a threshold.

---

# 4. Main theorem

## Theorem A (entropy-threshold stopping)
Suppose Assumptions R1-R6 hold. Then for each remaining budget `b`, there exists a threshold
\[
h^*(b) \in [0,\bar H] \cup \{\infty\}
\]
such that an optimal policy is given by
\[
\pi_b^*(H)
=
\begin{cases}
\texttt{STOP}, & H \le h^*(b),\\
\texttt{CONTINUE}, & H > h^*(b).
\end{cases}
\]
That is, the optimal stopping region is an interval of the form `[0,h^*(b)]`.

If in addition the effective cost of continuation weakly increases as budget shrinks, then the threshold is monotone in budget:
\[
h^*(b+1) \le h^*(b)
\qquad \text{for all } b.
\]
Equivalently, with more remaining budget, the controller is willing to continue even at lower entropy; with less remaining budget, it stops earlier.

---

# 5. Why these assumptions are natural for EGO

## R1 is exactly what entropy is for
Entropy is a proxy for residual ambiguity. If entropy is lower, the best current answer is more trustworthy on average.

## R2 says additional evidence preserves order
A state that is already more certain should not systematically produce worse future uncertainty after another reasoning / tool / delegate step.

## R3 encodes diminishing returns
This is the heart of the theorem. If continuation kept yielding the same marginal gain forever, threshold stopping would be much harder to justify. Diminishing entropy reduction is the formal version of "once the answer is almost clear, more work is not worth much."

## R4 and R5 create the budget dependence
Positive cost prevents endless continuation; extra budget creates option value. Together they produce the budget-aware threshold phenomenon.

---

# 6. Proof-ready structure

We now outline a proof that should survive formalization.

## Lemma 1: monotonicity of the value function
For every `b`, the value function `V_b(H)` is non-increasing in entropy `H`.

### Proof idea
Use induction on `b`.
- Base case: `V_0(H)=G(H,0)`, non-increasing by R1.
- Induction step: if `V_{b-1}` is non-increasing, then by R2 the continuation term
  \[
  C_b(H) := -c(b) + \int V_{b-1}(H') K_b(dH' \mid H)
  \]
  is also non-increasing in `H`.
- The maximum of two non-increasing functions is non-increasing.

So `V_b` is non-increasing.

## Lemma 2: monotonicity of the continuation advantage
Define
\[
\Delta_b(H) := \big[-c(b) + \int V_{b-1}(H') K_b(dH' \mid H)\big] - G(H,b).
\]
Then `\Delta_b(H)` is non-decreasing in `H`.

### Why non-decreasing is the right direction
- At low entropy, stopping is attractive.
- At high entropy, continuation becomes more attractive.
So the continuation-minus-stop gap should increase with entropy.

### Proof idea
- The continuation term is non-increasing in `H` by Lemma 1 and R2.
- But `G(H,b)` is also non-increasing in `H)`.
- To conclude monotonicity of the difference, we need a sharper comparison. This is where R3 enters: it ensures the continuation term deteriorates more slowly than the stopping payoff as entropy rises. Equivalently, the marginal advantage of an extra step is larger at higher entropy.
- Under the convexity/monotonicity structure in R3, this can be formalized using monotone comparative statics or a single-crossing argument.

## Lemma 3: interval structure of the stopping set
Because `\Delta_b(H)` is non-decreasing and continuous, the set
\[
\mathcal{S}_b := \{H : \Delta_b(H) \le 0\}
\]
is an interval of the form `[0,h^*(b)]`.

This yields the threshold rule immediately.

## Lemma 4: budget monotonicity
Under R5 and budget-dependent continuation cost, `\Delta_b(H)` weakly increases with `b`. Therefore the stopping threshold `h^*(b)` is weakly increasing in `b`.

This proves the second part of the theorem.

---

# 7. Single-crossing route: the cleanest formal proof tactic

The hardest step is Lemma 2. The most robust route is to frame the problem with a single-crossing property.

Define the action-value difference
\[
D(H,b) := Q_{cont}(H,b) - Q_{stop}(H,b).
\]
If we can show:
1. `D(H,b)` is continuous in `H`, and
2. `D(H,b)` has the single-crossing property in `H`, i.e., once it becomes nonnegative it stays nonnegative,

then the optimal policy is threshold-based by standard monotone-comparative-statics logic.

So the real proof burden is to derive single crossing from R1-R3.

This is likely cleaner in the final paper than trying to manipulate second derivatives directly.

---

# 8. A concrete corollary under a parametric entropy model

To make the theorem feel concrete, we can include a simple corollary.

## Corollary A1
Suppose the entropy dynamics satisfy
\[
H' = (1-\rho_b)H + \xi,
\]
where:
- `0 < \rho_b < 1`,
- `\xi` is mean-zero bounded noise independent of `H`,
- the stopping payoff is `G(H,b)=g_b-\alpha H` for some `\alpha>0`.

Then an optimal entropy-threshold stopping rule exists.

### Why this corollary is useful
- It is simple enough to verify directly.
- It gives a ready-made synthetic environment.
- It provides an intuitive sanity-check case for readers.

---

# 9. Connection to EGO's implemented gate

The theorem is stated in terms of scalar entropy `H`, whereas EGO-v1 uses
\[
(H_t, D_t, V_t)
\]
in its practical gate.

The intended paper narrative is:
1. **Theory core**: entropy alone already implies a budget-aware threshold stopping law.
2. **Practical extension**: disagreement and verifier confidence improve robustness when entropy is estimated from finite candidate pools.

So the implemented gate
\[
H_t \le h(B_t^{step}), \quad D_t \le d(B_t^{step}), \quad V_t \ge v(B_t^{step})
\]
should be presented as a robustified realization of the entropy-threshold principle, not as a separate theorem target.

That keeps the math clean and the method practical.

---

# 10. What to put in the paper main text

A concise main-text version can be:

> We first analyze a simplified budgeted stopping problem where the controller observes a scalar uncertainty statistic, instantiated as predictive entropy. Under monotone stopping utility, positive continuation cost, and diminishing expected entropy reduction, the optimal policy is threshold-based: for each remaining budget, there exists an entropy threshold below which the controller should stop. Moreover, this threshold increases with remaining budget, implying that budget-rich agents can justify further deliberation while budget-poor agents stop earlier.

This is the right level of density for AAAI.

---

# 11. What still needs to be written next

1. A short proposition deriving single crossing from a more explicit parametric entropy-reduction model.
2. A synthetic environment definition matching Corollary A1.
3. A notation table shared across formalization and theorem sections.
4. Optional: a second theorem bounding the utility loss from estimated entropy `\widehat H` instead of oracle entropy `H`.
