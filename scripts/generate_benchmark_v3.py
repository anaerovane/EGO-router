from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path('/Users/bytedance/Desktop/lunwen/aaai_agent_method')
OUT = ROOT / 'data' / 'realistic_mixed_task_benchmark_v3.json'

DIFF_CN = {'easy': '简单', 'medium': '中等', 'hard': '复杂'}
REASONING = {'easy': 1, 'medium': 2, 'hard': 3}
SPLITS = ['train'] * 36 + ['dev'] * 12 + ['test'] * 12


def meta(source: str, difficulty: str, idx: int) -> Dict[str, Any]:
    return {
        'source': source,
        'difficulty': difficulty,
        'difficulty_cn': DIFF_CN[difficulty],
        'expected_reasoning_steps': REASONING[difficulty],
        'split': SPLITS[idx - 1],
    }


items: List[Dict[str, Any]] = []


def add(item: Dict[str, Any]) -> None:
    items.append(item)


# ---------------- math: 60 ----------------
math_concepts = [
    ('budget-dependent stopping threshold', 'more budget means more willingness to continue'),
    ('value-based stopping', 'continuation should stop when net action value is non-positive'),
    ('candidate-pool posterior', 'candidate answers approximate the unavailable true posterior'),
    ('entropy threshold monotonicity', 'threshold decreases as remaining budget increases'),
    ('contextual-bandit routing', 'actions are scored under state-dependent features'),
    ('verifier confidence', 'low verifier confidence can keep the stop gate open'),
    ('disagreement', 'diverse channels can reveal hidden instability'),
    ('margin', 'top-two separation captures answer confidence differently from entropy'),
    ('oracle threshold extraction', 'dynamic programming provides a reference policy'),
    ('hybrid learned-plus-heuristic routing', 'stability and adaptability are traded off'),
]
math_templates = {
    'easy': [
        'Explain the role of {concept} in EGO.',
        'Why is {concept} important for the EGO controller?',
        'Give a concise explanation of {concept} in the EGO method.',
        'What does {concept} mean in the context of EGO?',
    ],
    'medium': [
        'Explain how {concept} affects both stopping and routing in EGO.',
        'Why does {concept} matter even when simpler heuristics might seem enough?',
        'Give a theorem-friendly intuition for {concept} in the EGO framework.',
        'Explain how {concept} interacts with the controller state and action scores.',
    ],
    'hard': [
        'Argue for and against simplifying EGO with a single rule around {concept}.',
        'Explain the main failure mode that could arise if {concept} is modeled poorly in EGO.',
        'Discuss how approximation error around {concept} could distort both stopping and routing decisions.',
        'Give a nuanced research-level explanation of why {concept} is useful but not sufficient by itself.',
    ],
}
for difficulty in ['easy', 'medium', 'hard']:
    base = 0 if difficulty == 'easy' else (20 if difficulty == 'medium' else 40)
    for local_idx in range(20):
        concept, reason = math_concepts[local_idx % len(math_concepts)]
        template = math_templates[difficulty][local_idx % len(math_templates[difficulty])]
        query = template.format(concept=concept)
        idx = base + local_idx + 1
        add({
            'id': f'math_{idx:03d}',
            'query': query,
            'task_type': 'math',
            'best_action': 'delegate:math',
            'reference_answer': None,
            'reference_points': [
                f'must correctly explain {concept}',
                reason,
                'should connect the concept to EGO rather than generic agent discussion',
                'should remain consistent with the project documents',
            ],
            'rubric': [
                'must be conceptually correct',
                'should connect to stopping, routing, or utility when relevant',
                'must avoid contradicting the current EGO project narrative',
            ],
            'evaluation_type': 'rubric_judge',
            'metadata': meta('project_derived_ai_assisted', difficulty, idx),
        })

# ---------------- calc: 60 ----------------
calc_easy_exprs = [
    ('0.08 + 0.75 / (1 + 1)', '0.455'), ('0.08 + 0.75 / (2 + 1)', '0.33'),
    ('0.08 + 0.75 / (3 + 1)', '0.2675'), ('0.08 + 0.75 / (4 + 1)', '0.22999999999999998'),
    ('0.08 + 0.75 / (5 + 1)', '0.20500000000000002'), ('0.08 + 0.75 / (6 + 1)', '0.18714285714285714'),
    ('1.0 - 0.2267', '0.7733'), ('1.0 - 0.2200', '0.78'), ('3.0 - 2.2', '0.7999999999999998'),
    ('3.0 - 1.0', '2.0'), ('3.0 - 0.7', '2.3'), ('2.29 - 1.98', '0.31000000000000005'),
    ('0.45 - 0.40', '0.04999999999999999'), ('0.12 * 15', '1.7999999999999998'), ('0.10 * 9', '0.9'),
    ('sqrt(16) + log(e)', '5.0'), ('abs(-3.4) + round(2.49)', '5.4'), ('pi - 3', '0.14159265358979312'),
    ('sqrt(81) / 3 + 2**3', '11.0'), ('exp(0) + cos(0) + sin(0)', '2.0'),
]
calc_medium_exprs = [
    ('0.55 * 0.8 + 0.20 * 0.3 + 0.10 * (1.0 - 0.4) + 0.05 * 3', '0.71'),
    ('0.65 * 0.9 + 0.25 * 0.2 + 0.20 * (1.0 - 0.3) + 0.50 * 0.6 + 0.25 * 0.55 + 0.10', '1.3125'),
    ('0.60 * 0.9 + 0.30 * 0.2 + 0.25 * (1.0 - 0.3) + 0.55 * 0.7 + 0.30 * 0.82 + 0.20', '1.406'),
    ('(0.6915 - 0.6889) / 0.6889', '0.003774133982'), ('(0.6915 + 0.6889) / 2', '0.6901999999999999'),
    ('(0.5944 - 0.2949)', '0.29950000000000004'), ('(0.2267 - 0.2200) / 0.2200', '0.030454545454545388'),
    ('(3.0 + 2.2 + 0.8) / 3', '2.0'), ('(3.0 + 0.7 + 0.5) / 3', '1.4000000000000001'),
    ('(3.0 + 0.9 + 0.3) / 3', '1.4000000000000001'), ('(2.6 + 1.2 + 0.9) / 3', '1.5666666666666667'),
    ('(0.16333333333333333 - 0.08) * 12', '0.9999999999999999'), ('(0.455 - 0.16333333333333333)', '0.2916666666666667'),
    ('round((0.71 + 1.3125) / 2, 4)', '1.0113'), ('(0.45 / 0.40) - 1', '0.125'),
    ('(0.455 - 0.20500000000000002) / 4', '0.0625'), ('(0.71 * 3 + 1.3125 * 2 + 1.406) / 6', '1.0268333333333333'),
    ('round(((0.7733 + 0.78) / 2) - 0.71, 4)', '0.0667'), ('((0.6915 + 0.6889 + 0.6759) / 3) - 0.6668', '0.01863333333333328'),
    ('((3.0 + 0.9 + 0.3) / 3) - ((2.6 + 1.2 + 0.9) / 3)', '-0.16666666666666652'),
]
calc_hard_exprs = [
    ('((0.60 * 0.9 + 0.30 * 0.2 + 0.25 * (1.0 - 0.3) + 0.55 * 0.7 + 0.30 * 0.82 + 0.20) - (0.65 * 0.9 + 0.25 * 0.2 + 0.20 * (1.0 - 0.3) + 0.50 * 0.6 + 0.25 * 0.55 + 0.10))', '0.09350000000000014'),
    ('((0.55 * 0.8 + 0.20 * 0.3 + 0.10 * (1.0 - 0.4) + 0.05 * 3) - 0.05) / (0.08 + 0.75 / (3 + 1))', '2.6542056074766354'),
    ('(0.6915 - 0.6354) / (4.0 - 2.25)', '0.03205714285714288'), ('(0.5944 - 0.2949) / 8.0', '0.037437500000000006'),
    ('(0.6889 - 0.6759) / (2.64 - 1.22)', '0.009154929577464818'), ('((3.0 - 0.7) - (3.0 - 0.5))', '-0.20000000000000018'),
    ('(0.45 * 40 - 0.40 * 40)', '1.9999999999999996'), ('((0.08 + 0.75 / (8 + 1)) - (0.08 + 0.75 / (1 + 1)))', '-0.2916666666666667'),
    ('((0.08 + 0.75 / (1 + 1)) / (0.08 + 0.75 / (8 + 1)))', '2.7857142857142856'), ('(1.406 - 1.3125) / 1.3125', '0.07123809523809535'),
    ('sqrt((3.0 - 2.2)**2 + (3.0 - 0.8)**2)', '2.340939982143925'), ('((0.5944 + 0.2949) / 2) / 0.6915', '0.6430224150397686'),
    ('((0.2267 - 0.2200) / 0.2267) * 100', '2.9554477273929995'), ('((3.0 + 2.2 + 0.8) / 3) - ((3.0 + 0.7 + 0.5) / 3)', '0.5999999999999999'),
    ('(0.455 - 0.20500000000000002) / (5 + 1 - (1 + 1))', '0.0625'), ('round((pi - 3) / 0.14159265358979312, 6)', '1.0'),
    ('((0.71 + 1.3125 + 1.406) / 3) / 0.71', '1.6077464788732394'), ('((0.31000000000000005 / 1.98) * 100)', '15.656565656565659'),
    ('((0.04999999999999999 / 0.40) * 100)', '12.499999999999996'), ('((0.29950000000000004 / 0.5944) * 100)', '50.38761709253029'),
]
for offset, difficulty, pairs in [(0, 'easy', calc_easy_exprs), (20, 'medium', calc_medium_exprs), (40, 'hard', calc_hard_exprs)]:
    for i, (expr, ans) in enumerate(pairs, start=1):
        idx = offset + i
        add({
            'id': f'calc_{idx:03d}',
            'query': f'Compute `{expr}`.',
            'task_type': 'calc',
            'best_action': 'tool:calculator',
            'reference_answer': ans,
            'reference_points': None,
            'rubric': None,
            'evaluation_type': 'numeric_exact',
            'metadata': {**meta('project_derived_ai_assisted', difficulty, idx), 'numeric_tolerance': 1e-9}
        })

# ---------------- search: 60 ----------------
search_targets = [
    ('src/integrations/learned_action_scorer.py', 'class LinUCBActionScorer', 'learned action scorer'),
    ('docs/12_mixed_task_benchmark.md', 'Mixed-Task Routing Benchmark', 'mixed-task routing benchmark design'),
    ('scripts/run_synthetic_theorem_a.py', 'Synthetic Theorem-A Validation', 'theorem-A synthetic stopping validation'),
    ('src/integrations/langchain_ego_adapter.py', 'class LangChainEGOAgent', 'LangChain-compatible orchestration wrapper'),
    ('docs/09_prototype_status.md', 'Prototype Status After First Synthetic Run', 'prototype status and oracle thresholds'),
    ('docs/01_formalization.md', 'EGO Formalization v1', 'formal problem definition'),
    ('src/integrations/experiment_runtime.py', 'OpenAICompatibleChatModel', 'real and mock runtime components'),
    ('src/integrations/ego_core.py', 'class EGOStoppingController', 'core stopping controller'),
    ('scripts/run_configurable_routing_experiment.py', 'build_real_agent', 'configurable mock/real runner'),
    ('data/realistic_mixed_task_benchmark_v3.json', 'difficulty_cn', '300-item realistic benchmark'),
    ('configs/experiment_mock.json', '"mode": "mock"', 'mock experiment config'),
    ('configs/experiment_real.json', '"mode": "real"', 'real experiment config'),
    ('README.md', 'Run mock / real 双模式实验', 'dual-mode experiment instructions'),
    ('docs/proposals/ego_orchestration_method_spec.md', 'Benchmark Construction Specification', 'benchmark construction rules'),
    ('docs/14_paper_method_section_draft.md', 'Paper Draft: Method Section', 'paper-aligned method draft'),
]
search_q_templates = {
    'easy': [
        'Which file defines the {desc}?',
        'Which file should I open to find the {desc}?',
        'Where in this project is the {desc} defined?',
        'Which file most directly contains the {desc}?',
    ],
    'medium': [
        'Which file should be searched first to understand the {desc}?',
        'If I want to inspect the {desc}, which file is the best entry point?',
        'Which file most directly explains or implements the {desc}?',
        'Where should a developer look first to study the {desc}?',
    ],
    'hard': [
        'Find the single file that best connects the project narrative to the {desc}.',
        'Which file would provide the strongest direct evidence for the {desc}?',
        'If you had to verify the {desc} with one file only, which file would you inspect?',
        'Which file most directly operationalizes the {desc} in this repository?',
    ],
}
for difficulty in ['easy', 'medium', 'hard']:
    offset = {'easy': 0, 'medium': 20, 'hard': 40}[difficulty]
    for i in range(20):
        path, evidence, desc = search_targets[i % len(search_targets)]
        query = search_q_templates[difficulty][i % 4].format(desc=desc)
        idx = offset + i + 1
        add({
            'id': f'search_{idx:03d}',
            'query': query,
            'task_type': 'search',
            'best_action': 'tool:search',
            'reference_answer': path,
            'reference_points': [evidence],
            'rubric': None,
            'evaluation_type': 'substring_or_evidence_match',
            'metadata': meta('project_corpus_ai_assisted', difficulty, idx),
        })

# ---------------- code: 60 ----------------
code_targets = [
    ('src/integrations/langchain_ego_adapter.py', ['ExpertSpec', '_score_actions', '_execute_action'], 'adding a new delegate expert action'),
    ('src/integrations/learned_action_scorer.py', ['EGOFeatureBuilder', 'build'], 'changing contextual-bandit features'),
    ('src/integrations/ego_core.py', ['EGOStoppingController', 'entropy_threshold', 'decide'], 'changing entropy-threshold stopping logic'),
    ('scripts/run_configurable_routing_experiment.py', ['build_real_agent', 'build_mock_agent', 'mode'], 'swapping mock and real experiment modes'),
    ('src/integrations/experiment_runtime.py', ['LocalCorpusSearchTool', '_score_text'], 'changing local corpus search behavior'),
    ('scripts/mixed_task_routing_benchmark.py', ['BenchmarkTask', 'TASKS'], 'editing the old controlled benchmark tasks'),
    ('docs/proposals/ego_orchestration_method_spec.md', ['Benchmark item schema', 'Evaluation types'], 'updating benchmark schema or evaluation types'),
    ('README.md', ['Run mock / real 双模式实验'], 'changing usage instructions'),
    ('configs/experiment_real.json', ['costs', 'real', 'search_corpus_dir'], 'changing real experiment config behavior'),
    ('src/integrations/experiment_runtime.py', ['LLMJudgeVerifier', '_build_prompt'], 'changing judge prompt behavior'),
    ('src/integrations/ego_core.py', ['CandidatePosteriorEstimator', '_softmax'], 'modifying posterior estimation logic'),
    ('src/integrations/langchain_ego_adapter.py', ['_default_tool_relevance', '_default_expert_relevance'], 'adjusting keyword-based relevance estimation'),
    ('scripts/run_configurable_routing_experiment.py', ['evaluate_agent', 'print_metrics'], 'adding new aggregate metrics'),
    ('src/integrations/experiment_runtime.py', ['SafeCalculatorTool', '_eval_node'], 'extending calculator parsing capability'),
    ('scripts/run_configurable_routing_experiment.py', ['load_tasks', 'QueryTask'], 'preserving more benchmark metadata at load time'),
]
code_q_templates = {
    'easy': [
        'If you want to work on {desc}, which file should you edit first?',
        'Which file is the first place to inspect for {desc}?',
        'Where should a developer start editing for {desc}?',
        'Which module is the primary entry point for {desc}?',
    ],
    'medium': [
        'If you were refactoring {desc}, which file would you change before touching anything else?',
        'Which file most directly owns the implementation for {desc}?',
        'Where should a code reviewer look first to validate changes for {desc}?',
        'Which file would be the cleanest first integration point for {desc}?',
    ],
    'hard': [
        'Describe the single most important file to inspect if {desc} needs a design-level refactor.',
        'Which file most tightly couples configuration, runtime logic, or metrics to {desc}?',
        'If {desc} were failing in subtle ways, which file should be audited first and why?',
        'Which file would you change first if you needed to redesign {desc} without breaking the whole pipeline?',
    ],
}
for difficulty in ['easy', 'medium', 'hard']:
    offset = {'easy': 0, 'medium': 20, 'hard': 40}[difficulty]
    for i in range(20):
        path, points, desc = code_targets[i % len(code_targets)]
        query = code_q_templates[difficulty][i % 4].format(desc=desc)
        idx = offset + i + 1
        add({
            'id': f'code_{idx:03d}',
            'query': query,
            'task_type': 'code',
            'best_action': 'delegate:code',
            'reference_answer': path,
            'reference_points': points,
            'rubric': [
                'must identify the most relevant file',
                'should mention at least one key class, function, or config field when useful',
                'must stay aligned with the current repository structure',
            ],
            'evaluation_type': 'reference_points_judge',
            'metadata': meta('project_corpus_ai_assisted', difficulty, idx),
        })

# ---------------- think: 60 ----------------
think_concepts = [
    ('the difference between stopping and routing in EGO', 'stopping decides whether to continue, routing decides which action to take'),
    ('why this project is a method paper prototype rather than a benchmark project', 'the core contribution is the orchestration method, with benchmark as validation scaffold'),
    ('why a hybrid learned-plus-heuristic routing score can be more stable than a purely learned score', 'the heuristic term stabilizes learning under sparse or noisy rewards'),
    ('why first-action accuracy is a useful metric', 'it directly measures routing quality and is not identical to final reward'),
    ('the role of the candidate pool', 'it supports posterior approximation and uncertainty estimation'),
    ('why cost-aware utility matters more than plain answer quality for orchestration', 'the controller optimizes information gain under resource constraints'),
    ('why benchmark construction is harder here than for standard QA', 'both final answers and action choices require supervision'),
    ('why AI-assisted benchmark construction must be described carefully', 'it is efficient but can introduce labeling bias'),
    ('why project-centric realistic benchmarks are useful', 'they bridge toy control and real task structure'),
    ('what the configurable runner adds beyond the old mixed-task script', 'it separates mock and real modes and prepares for real experiments'),
    ('why synthetic and realistic benchmarks are both needed', 'they support different claims: theory alignment versus realism'),
    ('why best_action labels matter even if final reward is measured', 'they isolate routing quality'),
    ('why open-ended tasks need rubrics rather than exact string matching', 'many valid phrasings can be correct'),
    ('why current search tasks are local-corpus rather than web search', 'this improves reproducibility and fits the current environment'),
    ('why stop-reason distributions are useful diagnostics', 'they reveal how the controller terminates, not just how well it scores'),
]
think_q_templates = {
    'easy': [
        'Briefly explain {concept}.',
        'Give a concise explanation of {concept}.',
        'Summarize {concept} in one short paragraph.',
        'In simple terms, explain {concept}.',
    ],
    'medium': [
        'Explain {concept} and why it matters for evaluating EGO.',
        'Give a clear project-level explanation of {concept}.',
        'Why does {concept} matter in this benchmark and method setup?',
        'Explain {concept} without reducing it to a generic ML statement.',
    ],
    'hard': [
        'Give a nuanced research-style explanation of {concept}, including one limitation or caveat.',
        'Explain {concept} in a way that would be defensible in a method paper discussion section.',
        'Discuss {concept} carefully, including why a simplistic explanation would be misleading.',
        'Provide a balanced explanation of {concept}, including both its value and one important caution.',
    ],
}
for difficulty in ['easy', 'medium', 'hard']:
    offset = {'easy': 0, 'medium': 20, 'hard': 40}[difficulty]
    for i in range(20):
        concept, point = think_concepts[i % len(think_concepts)]
        query = think_q_templates[difficulty][i % 4].format(concept=concept)
        idx = offset + i + 1
        add({
            'id': f'think_{idx:03d}',
            'query': query,
            'task_type': 'think',
            'best_action': 'think',
            'reference_answer': None,
            'reference_points': [
                point,
                'the answer should remain aligned with the current project materials',
                'the explanation should emphasize orchestration-specific relevance',
            ],
            'rubric': [
                'must be conceptually coherent',
                'should avoid overclaiming beyond the current benchmark maturity',
                'should be concise for easy items and more nuanced for hard items',
            ],
            'evaluation_type': 'rubric_judge',
            'metadata': meta('project_derived_ai_assisted', difficulty, idx),
        })

assert len(items) == 300, len(items)
counts = {}
for item in items:
    counts[item['task_type']] = counts.get(item['task_type'], 0) + 1
assert counts == {'math': 60, 'calc': 60, 'search': 60, 'code': 60, 'think': 60}, counts

OUT.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'wrote {len(items)} items to {OUT}')
print(counts)
