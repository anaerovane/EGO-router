# Learned Action Scoring for EGO Routing

## Goal
Upgrade the routing layer from purely hand-crafted action scores to a learned contextual-bandit style scorer that fits the paper's regret / online-learning story.

---

## Files added
- `src/integrations/learned_action_scorer.py`
- `scripts/langchain_ego_learned_scoring_demo.py`

---

## What is implemented

### 1. `EGOFeatureBuilder`
Builds linear feature vectors for actions using:
- entropy
- margin
- disagreement
- verifier confidence
- remaining steps
- action cost
- relevance
- prior relevance
- action-type indicators
- interaction terms

This is the feature layer needed for a contextual-bandit view of routing.

### 2. `LinUCBActionScorer`
Implements a simple per-action linear UCB scorer:
- ridge-regularized linear model per action
- exploration bonus from uncertainty
- online update from observed reward signal

This is not yet a production-quality learner, but it is exactly the kind of module that supports a clean regret-style discussion in the paper.

### 3. Optional integration into `LangChainEGOAgent`
The adapter now supports:
- `use_learned_action_scorer=True`
- optional `learned_scorer`
- optional `learned_gain_mix`

So the routing score can now be a **hybrid** of:
- learned predicted reward
- exploration bonus
- heuristic estimated gain
- action cost

This hybrid design is deliberate: it makes the prototype stable before we have real interaction logs.

---

## Why hybrid scoring is a good intermediate step
A purely learned scorer with no prior often over-explores in tiny demos. A purely heuristic scorer is easier to control but weaker as a paper contribution.

The hybrid score is a practical middle ground:
\[
\text{score}(a)
=
\widehat r_{learned}(a)
+
\text{bonus}(a)
+
\lambda \cdot \widehat{gain}_{heuristic}(a)
-
\text{cost}(a)
\]

This keeps the system usable while preserving a principled path toward learned routing.

---

## Reward signal used right now
After executing an action, the current prototype computes a lightweight reward signal based on:
- verifier-confidence improvement;
- entropy reduction after adding the new candidate.

This is only a first proxy for realized value of information.

A stronger future version should use:
- downstream answer correctness improvement;
- calibrated utility gain;
- cost-aware reward shaping.

---

## Relation to the paper
This module is the first implementation step toward the paper's routing story:
- Theorem A supports stopping.
- Theorem B supports robustness to uncertainty estimation.
- The learned scorer supports a later **regret / online-learning** extension for action selection.

So this component is especially relevant if the paper evolves from
"threshold stopping + heuristic routing"
into
"threshold stopping + learned budget-aware orchestration."

---

## Current limitation
The current learned demo is still a scaffold:
- it uses synthetic mock experts/tools;
- it updates online from a simple proxy reward;
- it is not yet trained on a realistic distribution of tasks.

So right now, it should be viewed as:
> a proof-of-implementation for learned routing,
not yet a strong empirical result.

---

## Immediate next upgrade
The next meaningful improvement would be:
1. log real trajectories from mixed synthetic tasks;
2. train the scorer offline or online on those trajectories;
3. compare learned routing against heuristic routing in a controlled benchmark.

That would make the routing section much more paper-ready.
