# AAAI Agent Method Project Todo

Project: Entropy-Gated Orchestration (EGO) for Budgeted LLM Agents  
Goal: Develop a mathematically grounded agent-orchestration method suitable for an AAAI submission, with provable stopping behavior, principled delegation/tool-use decisions, and convincing experiments.

## North-star deliverable
- A method paper centered on a budget-aware meta-controller that decides when to think, use tools, delegate, or stop.
- Core ingredients: formal problem definition, algorithm, at least 1-2 clean theorems, synthetic validation, realistic benchmark evidence, and a paper-ready narrative.

---

## Phase 0. Project framing
- [x] Pick the main direction: uncertainty-aware orchestration / stopping.
- [x] Name the method family: EGO (Entropy-Gated Orchestration).
- [ ] Lock the exact problem scope for v1:
  - single final-answer tasks
  - finite token / latency / step budget
  - actions = think / tool / delegate / stop
  - belief state from sampled candidates + verifier
- [ ] Decide what will be in the first paper and what will be deferred.

## Phase 1. Formalization
- [ ] Define the sequential decision problem rigorously.
- [ ] Define state, action, observation, transition, budget, and stopping rule.
- [ ] Define posterior approximation and uncertainty statistics.
- [ ] Define the objective as utility minus budget costs.
- [ ] Identify the smallest theorem-friendly assumptions.
- [ ] Write a paper-ready notation table.

## Phase 2. Algorithm design
- [ ] Specify EGO-v1 scoring function.
- [ ] Define value-of-information approximation for THINK / TOOL / DELEGATE.
- [ ] Define entropy gate and value-based stopping rule.
- [ ] Decide whether v1 uses linear scoring, bandit learning, or a small learned critic.
- [ ] Write pseudocode and identify implementation interfaces.
- [ ] List variants: threshold-only, VOI-only, full EGO.

## Phase 3. Theory
- [ ] Theorem A: existence of a budget-dependent threshold stopping policy under monotonicity / diminishing returns assumptions.
- [ ] Theorem B: suboptimality bound for approximate stopping with imperfect posterior estimates.
- [ ] Optional Theorem C: regret bound for online orchestration (contextual bandit view).
- [ ] Define assumptions clearly and keep them as weak as possible.
- [ ] Prepare proof sketches before full proofs.
- [ ] Design synthetic environments that directly test theorem predictions.

## Phase 4. Experimental protocol
- [ ] Build a synthetic oracle environment where information gain and costs are controllable.
- [ ] Build a lightweight real-task evaluation setup.
- [ ] Choose task families:
  - factual QA / retrieval-heavy
  - math / symbolic
  - tool-necessary tasks
  - mixed routing tasks for delegation
- [ ] Define metrics:
  - accuracy / success
  - total tokens
  - latency
  - number of tool/delegate actions
  - utility under cost tradeoff
  - calibration / stopping quality
- [ ] Define ablations for each component.

## Phase 5. Baselines
- [ ] Fixed-depth think-then-answer.
- [ ] Always-think-until-budget.
- [ ] Heuristic uncertainty threshold.
- [ ] Tool-first / delegate-first heuristics.
- [ ] Simple UCB / bandit router.
- [ ] If feasible, compare against a strong workflow baseline.

## Phase 6. Implementation plan
- [ ] Create simulator for tasks, tools, experts, and budget transitions.
- [ ] Implement posterior estimation from candidate answers.
- [ ] Implement verifier interface.
- [ ] Implement EGO controller.
- [ ] Implement logging for action traces and cost traces.
- [ ] Implement experiment scripts and plotting utilities.

## Phase 7. Paper writing
- [ ] Draft title / abstract / intro.
- [ ] Draft related work with focus on agent orchestration, stopping, budgeted planning, and uncertainty-aware control.
- [ ] Draft method section from formalization doc.
- [ ] Draft theory section from proved results.
- [ ] Draft experiment section from results.
- [ ] Create figures:
  - framework diagram
  - stopping-threshold intuition
  - cost-performance Pareto plot
  - action allocation histograms
- [ ] Draft discussion / limitations / broader impact.

## Immediate next actions
1. Write a rigorous mathematical formulation of the EGO problem.
2. Turn that formulation into a theorem-ready proposition list.
3. Freeze EGO-v1 algorithmic choices.
4. Then move to synthetic environment design.

## Working principles
- Prefer the smallest clean formulation that can support theorem + experiment.
- Avoid overengineering the first version.
- Every component should answer one of these questions:
  - Why continue?
  - Which action is worth paying for?
  - When should the agent stop?
- Keep a clean separation between formal core and engineering extensions.

---

## Progress log
- 2026-05-22: Created project workspace at `lunwen/aaai_agent_method`.
- 2026-05-22: Wrote roadmap into `todo.md`.
- 2026-05-22: Finished rigorous formalization in `docs/01_formalization.md`.
- 2026-05-22: Finished theorem scaffold in `docs/02_theory_scaffold.md`.
- 2026-05-22: Froze EGO-v1 algorithm in `docs/03_algorithm_v1.md`.
- 2026-05-22: Drafted experiment plan in `docs/04_experimental_blueprint.md`.
- 2026-05-22: Started first main theorem draft in `docs/05_theorem_A_draft.md`.
- 2026-05-22: Refined Theorem A into a proof-ready entropy-threshold result in `docs/06_theorem_A_refined.md`.
- 2026-05-22: Designed the first theory-aligned synthetic environment in `docs/07_synthetic_env_for_theorem_A.md`.
- 2026-05-22: Drafted Theorem B on stopping utility loss under uncertainty-estimation error in `docs/08_theorem_B_draft.md`.
- 2026-05-22: Implemented the first runnable synthetic prototype and recorded initial findings in `docs/09_prototype_status.md`.
- 2026-05-22: Added a LangChain-compatible EGO adapter layer in `src/integrations/` with a runnable example and integration doc.
- 2026-05-22: Upgraded the LangChain adapter to multi-action scoring over think vs multiple tools, with a runnable multi-tool demo.
- 2026-05-22: Added delegate/expert-agent actions to the LangChain-compatible EGO adapter, completing think/tool/delegate/stop coverage.
- 2026-05-22: Added a contextual-bandit style learned action-scoring prototype (`LinUCBActionScorer`) and integrated it into the LangChain-compatible EGO adapter.
- 2026-05-22: Added a mixed-task routing benchmark to compare heuristic vs learned EGO routing across math/calc/search/code/think task types.
