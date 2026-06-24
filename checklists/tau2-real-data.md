# Tau2 Real Data Checklist

> 用于约束“先产出 1 条 tau2 真实 state-level、action-scored 样本”的需求质量，避免重新滑回伪数据导出或训练准备。

Purpose: 审查当前数据构造要求是否完整、清晰、可验收，并能直接指导执行。
Created: 2026-05-26
Source scope: `docs/16_handoff_prompt_for_data_construction.md`, `docs/15_data_status_and_paths.md`, `README.md`, `/private/tmp/tau2-bench_escalated/docs/evaluation.md`, `/private/tmp/tau2-bench_escalated/pyproject.toml`
Depth: Standard
Actor/Timing: Author before implementation; reviewer before accepting the first “real sample” deliverable

## Requirement Completeness

- [ ] CHK001 Are the “minimum viable deliverable” requirements explicitly limited to **one real tau2 sample before any training or dataset mixing**? [Completeness, Handoff §你的唯一任务, Handoff §Phase 1, README §重要状态说明]
- [ ] CHK002 Are all mandatory fields for a “real sample” fully enumerated, rather than partially implied? [Completeness, Handoff §什么算“真实训练数据”, Data Status §十]
- [ ] CHK003 Are the required inputs for the first tau2 run documented, including benchmark path, domain, task identifier, and target decision step selection rule? [Completeness, Handoff §当前项目里的 benchmark / 数据来源, Handoff §Step B, Data Status §九]
- [ ] CHK004 Are the required state components defined for capture at a decision point, including query, history summary, candidate pool, uncertainty metrics, and remaining budget? [Completeness, Handoff §什么算“真实训练数据”, Data Status §九]
- [ ] CHK005 Are the requirements for `available_actions` defined as **state-aligned and environment-aligned**, not a fixed global template? [Completeness, Data Status §Step 3, Handoff §Step C]
- [ ] CHK006 Are the requirements for `action_scores` defined for **all candidate actions**, not only the selected best action? [Completeness, Handoff §正确的数据构造目标格式, Data Status §Step 5]
- [ ] CHK007 Does the work definition explicitly require a reproducible scoring method description alongside the sample itself? [Completeness, Handoff §你交付时必须写清楚的内容, Data Status §Step 4]
- [ ] CHK008 Are the output artifact requirements complete, including sample file path, collection script path, source benchmark/domain/task/step, and remaining approximation notes? [Completeness, Handoff §你交付时必须写清楚的内容]

## Requirement Clarity

- [ ] CHK009 Is “真实运行” defined with enough precision to distinguish a genuine tau2 episode from raw task-file conversion or template filling? [Clarity, Handoff §Step A, Data Status §当前结论, README §重要状态说明]
- [ ] CHK010 Is the term “decision point” operationally defined so two readers would identify the same capture boundary in an episode? [Clarity, Handoff §Step B, Data Status §Step 2] 
- [ ] CHK011 Is the candidate-pool requirement explicit about acceptable sources for the first sample, such as draft / think-refine / tool-augmented candidates? [Clarity, Handoff §Step B, Data Status §九]
- [ ] CHK012 Is the requirement for “真实可选动作集合” clear about whether read-only tools, write tools, transfer actions, and `stop`/`think` must all be considered when available? [Clarity, Handoff §Step C, Data Status §Step 3]
- [ ] CHK013 Is the acceptable proxy for first-pass `action_scores` sufficiently specified so “可靠近似打分” is not left subjective? [Ambiguity, Handoff §Step D, Data Status §Step 4]
- [ ] CHK014 Is “不要继续推进伪数据扩充或训练” translated into explicit exclusion boundaries for scripts, outputs, and success criteria? [Clarity, Handoff §第一原则, Handoff §当前仓库里哪些东西不要继续当主数据用]
- [ ] CHK015 Is the environment-readiness requirement specific about what must be recorded when tau2 cannot run (Python version, missing deps, API key, CLI/import issue)? [Clarity, Handoff §Step A]

## Requirement Consistency

- [ ] CHK016 Do the handoff, README, and data-status documents consistently define the current real-data count as zero and forbid treating historical `jsonl` files as the main dataset? [Consistency, Handoff §第一原则, README §重要状态说明, Data Status §当前结论]
- [ ] CHK017 Are the sample-format requirements consistent between the “real sample” JSON schema and the later SFT-conversion description, especially around `best_action`, `action_scores`, and `should_stop` derivation? [Consistency, Data Status §十, Data Status §十二]
- [ ] CHK018 Do the requirements consistently separate **reference trajectory** from **supervision target**, avoiding any implication that tau2 `evaluation_criteria.actions` can be used directly as router labels? [Consistency, Handoff §第一原则, tau2 Evaluation TL;DR]
- [ ] CHK019 Are the environment constraints consistent with the actual tau2 packaging requirements, including the stated Python version range? [Consistency, Handoff §Step A, tau2 pyproject `requires-python`]
- [ ] CHK020 Do the required deliverables align with the stated priority order “1 real sample first, scale later,” without mixed signals toward bulk generation? [Consistency, Handoff §Phase 1-3, Data Status §十三]

## Acceptance Criteria Quality

- [ ] CHK021 Can acceptance of the first sample be decided objectively from documented checks rather than reviewer intuition? [Acceptance Criteria, Handoff §Phase 2]
- [ ] CHK022 Is there an objective rule for validating `best_action = argmax(action_scores)` in the final artifact? [Measurability, Handoff §正确的数据构造目标格式]
- [ ] CHK023 Are pass/fail criteria defined for the environment-readiness stage before data collection begins? [Acceptance Criteria, Handoff §Step A, Gap]
- [ ] CHK024 Is there a measurable requirement for proving that the sample came from a real episode, such as trajectory/log provenance or replayable collection steps? [Measurability, Handoff §Phase 2, Gap]
- [ ] CHK025 Are success criteria explicit for the scoring method write-up, including whether it is full utility or one-step proxy and what approximation remains? [Acceptance Criteria, Handoff §你交付时必须写清楚的内容]

## Scenario Coverage

- [ ] CHK026 Are requirements defined for the happy path where tau2 runs successfully and yields at least one usable decision point? [Coverage, Handoff §Phase 1, Data Status §Step 1]
- [ ] CHK027 Are requirements defined for the blocked path where tau2 cannot import or execute, including what diagnostic evidence must be captured? [Coverage, Exception Flow, Handoff §Step A]
- [ ] CHK028 Are requirements defined for the path where the episode runs but no meaningful candidate pool exists yet, including how a minimal pool may be constructed without reverting to fake labels? [Coverage, Exception Flow, Handoff §Step B]
- [ ] CHK029 Are requirements defined for the path where multiple actions appear plausible, including how ties or near-ties in `action_scores` should be documented? [Coverage, Gap]
- [ ] CHK030 Are requirements defined for the follow-up path from “1 sample” to “10/100 samples,” while still preventing premature scaling before the first sample passes acceptance? [Coverage, Handoff §Phase 3, Data Status §十三]

## Edge Case Coverage

- [ ] CHK031 Are boundary conditions specified for tasks whose correct behavior is immediate refusal or no-op, where the real best action may be `stop`? [Edge Case, tau2 Evaluation §worked example, Handoff §什么算“真实训练数据”]
- [ ] CHK032 Are requirements defined for cases where the current state exposes only a subset of domain tools, rather than the full airline tool universe? [Edge Case, Data Status §Step 3, Handoff §Step C]
- [ ] CHK033 Are requirements defined for steps with low entropy but still non-trivial action alternatives, so stopping criteria and routing criteria do not conflict? [Edge Case, Handoff §Step D, Gap]
- [ ] CHK034 Are requirements defined for missing or weak verifier signals, including how `verifier_confidence` should be represented and how scoring should degrade gracefully? [Edge Case, Data Status §Step 4, Gap]

## Dependencies & Assumptions

- [ ] CHK035 Are external prerequisites documented as dependencies rather than hidden assumptions, especially Python 3.12+, package installation method, and any API-key needs? [Dependencies, Handoff §Step A, tau2 README §Quick Start, tau2 pyproject]
- [ ] CHK036 Is the assumption that tau2 airline is the first target domain explicitly justified and bounded, rather than left as an implicit preference? [Assumption, Handoff §推荐的最短执行路线]
- [ ] CHK037 Are assumptions about how utility or proxy reward can be estimated from a one-step transition documented and reviewable? [Assumption, Handoff §Step D, Data Status §Step 4]
- [ ] CHK038 Are dependencies between collection script, runtime logs, sample file, and acceptance review documented so the first sample can be reproduced later? [Dependencies, Handoff §你交付时必须写清楚的内容, Gap]

## Ambiguities & Conflicts

- [ ] CHK039 Is it unambiguous whether the first deliverable must be generated from the repo’s existing EGO runtime, a tau2-native runner, or any reproducible collector that operates on the real environment? [Ambiguity, Gap]
- [ ] CHK040 Is it unambiguous how much approximation is acceptable in the first `action_scores` pass before a sample stops qualifying as “real collected data”? [Ambiguity, Handoff §Step D]
- [ ] CHK041 Is it unambiguous what evidence distinguishes “minimal candidate-pool construction” from the forbidden practice of fabricating task-level pseudo labels? [Ambiguity, Handoff §第一原则, Handoff §Step B]
- [ ] CHK042 Are conflicts resolved between the desire for a “minimal closed loop” and the need for enough provenance to verify the sample is truly real-collected? [Conflict, Handoff §Phase 1-2, Gap]
