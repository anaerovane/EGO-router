# EGO-router Project Specification

## Project name

**EGO-router**: Entropy-Gated Orchestration for Budget-Aware LLM Agents.

## Goal

EGO-router studies a budget-aware meta-controller for LLM agents. The controller decides, at each step, whether an agent should:

- `THINK`: perform another internal reasoning step;
- `TOOL(m)`: call an external tool;
- `DELEGATE(k)`: ask a specialist/expert agent;
- `STOP`: terminate and return the current best answer.

The central objective is to maximize task utility under resource constraints such as step budget, token cost, latency cost, and risk cost.

## Core idea

EGO summarizes the current agent trajectory into compact uncertainty and budget features:

- predictive entropy;
- answer margin;
- disagreement across candidates, tools, or experts;
- verifier confidence;
- remaining step/token/latency budget.

The method combines:

1. **Budget-aware stopping**: stop when uncertainty is low enough for the remaining budget, or when no continuation action has positive net value.
2. **Value-based routing**: if continuing is worthwhile, select the action with the highest estimated value of information minus cost.
3. **Learned routing extension**: optionally replace or augment heuristic action scoring with contextual-bandit or SFT-trained router models.

## Repository layout

```text
README.md                         # Project overview and status
todo.md                           # Roadmap and progress log
PROJECT_SPEC.md                   # This concise project specification
.env.example                      # Environment variable template without secrets

src/
  envs/                           # Synthetic environments
  solvers/                         # DP oracle solver
  policies/                        # Baseline and EGO threshold policies
  integrations/                    # EGO core, LangChain adapter, runtime/evaluator
  training/                        # Router SFT utilities

scripts/
  run_synthetic_theorem_a.py       # First theorem-aligned synthetic validation
  run_synthetic_sweeps.py          # Systematic synthetic sweep runner
  summarize_synthetic_sweep.py     # Sweep summary helper
  run_configurable_routing_experiment.py
  run_baseline_matrix.py
  export_router_sft_data.py
  convert_*_to_router_sft.py

docs/
  01_formalization.md
  03_algorithm_v1.md
  05_theorem_A_draft.md
  06_theorem_A_refined.md
  08_theorem_B_draft.md
  14_paper_method_section_draft.md
  15_data_status_and_paths.md
  17_synthetic_sweep_findings.md

data/                              # Benchmark task files
configs/                           # Experiment configs
training/                          # LLaMA-Factory SFT starter pack
checklists/                        # Data/release checklists
```

## Current status

Implemented:

- theorem-aligned scalar entropy environment;
- dynamic-programming oracle;
- baseline stopping policies;
- minimal EGO entropy gate;
- systematic synthetic sweep runner;
- EGO core controller;
- LangChain-compatible adapter for think/tool/delegate/stop;
- contextual-bandit style learned action scorer;
- mixed-task routing benchmark and evaluator;
- router SFT data conversion and LLaMA-Factory starter configuration.

Not yet finalized:

- paper-ready theorem proofs;
- final train/dev/test split for real router SFT data;
- full RL/GRPO training;
- final paper figures and benchmark tables.

## Data status

The repository includes benchmark-style JSON files and warm-start SFT examples. Real state-level tau2 decision-point samples are documented in `docs/15_data_status_and_paths.md`, but the final formal router SFT dataset is not yet frozen.

Do not treat generated `router_sft_*.jsonl` files as the final real training dataset unless they have been explicitly validated and registered.

## Security / secret handling

This repository intentionally excludes local `.env` files and generated outputs. Use `.env.example` as a template and set API keys locally.

Never commit:

- `.env`;
- API keys;
- local Claude settings;
- private benchmark dumps;
- generated logs containing request payloads or credentials.
