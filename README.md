# EGO-router

**Entropy-Gated Orchestration for Budget-Aware LLM Agents**

EGO-router is a research codebase for studying **budget-aware orchestration of LLM agents**. It focuses on a high-level controller that decides, at each agent step, whether to:

- `THINK` — run another internal reasoning/refinement step;
- `TOOL(m)` — invoke an external tool;
- `DELEGATE(k)` — ask a specialist/expert agent;
- `STOP` — terminate and return the current best answer.

The project is not a production agent platform. It is a method-oriented research prototype for formalizing, analyzing, and experimentally validating agent orchestration policies under step/token/latency budgets.

---

## 1. Research goal

Most LLM-agent workflows hard-code a fixed pattern such as "reason → search → answer", or rely on prompting for tool-use decisions. EGO-router instead treats orchestration itself as a **budgeted sequential decision problem**.

The main questions are:

1. When should an agent keep thinking?
2. When is a tool call worth its cost?
3. When should the task be delegated to an expert/sub-agent?
4. When should the agent stop and answer?

The target objective is cost-aware utility:

```text
final task reward - token cost - latency cost - step cost - risk cost
```

---

## 2. Core idea

EGO summarizes the current agent trajectory into compact uncertainty and budget features:

- predictive entropy;
- top-1/top-2 answer margin;
- disagreement across candidates, tools, reasoning paths, or experts;
- verifier confidence;
- remaining step/token/latency budget.

It combines two control layers:

### 2.1 Budget-aware stopping

The controller stops when uncertainty is sufficiently low for the remaining budget, or when no continuation action has positive net value.

A minimal entropy gate uses a budget-dependent threshold:

```text
STOP if H_t <= h(B_t)
```

where a simple implementation is:

```text
h(B_t) = h0 + alpha_h / (B_t + 1)
```

This makes the controller more willing to continue when budget is abundant and more willing to stop when budget is scarce.

### 2.2 Value-based routing

If continuing is worthwhile, the controller scores continuation actions by an approximate value of information minus action cost:

```text
score(action) = estimated information gain - cost(action)
```

The action space is:

```text
{THINK, TOOL(m), DELEGATE(k), STOP}
```

### 2.3 Learned routing extension

The repository also contains a contextual-bandit style learned action scorer and a router SFT starter pack. Full RL/GRPO training is not implemented yet, but the formulation is compatible with future agentic-RL training.

---

## 3. Repository structure

```text
.
├── README.md
├── PROJECT_SPEC.md
├── todo.md
├── .env.example
├── requirements.txt
│
├── src/
│   ├── envs/                 # Synthetic entropy environment
│   ├── solvers/              # Dynamic-programming oracle
│   ├── policies/             # Baseline and EGO stopping policies
│   ├── integrations/         # EGO core, LangChain-style adapter, evaluator/runtime
│   └── training/             # Router SFT utilities
│
├── scripts/
│   ├── run_synthetic_theorem_a.py
│   ├── run_synthetic_sweeps.py
│   ├── summarize_synthetic_sweep.py
│   ├── mixed_task_routing_benchmark.py
│   ├── run_configurable_routing_experiment.py
│   ├── run_baseline_matrix.py
│   └── convert_*_to_router_sft.py
│
├── docs/
│   ├── 01_formalization.md
│   ├── 03_algorithm_v1.md
│   ├── 05_theorem_A_draft.md
│   ├── 06_theorem_A_refined.md
│   ├── 08_theorem_B_draft.md
│   ├── 14_paper_method_section_draft.md
│   ├── 15_data_status_and_paths.md
│   └── 17_synthetic_sweep_findings.md
│
├── configs/                  # Mock/real experiment configs
├── data/                     # Mixed-task benchmark JSON files
├── outputs/                  # Core local experiment/data artifacts
├── training/                 # LLaMA-Factory router SFT starter pack
└── checklists/               # Data/release validation checklists
```

---

## 4. Current status

Implemented:

- theorem-aligned scalar entropy environment;
- dynamic-programming oracle;
- fixed-depth, fixed-threshold, immediate-stop, never-stop, and oracle baselines;
- minimal EGO entropy-gate policy;
- synthetic sweep runner and result summaries;
- framework-independent EGO core controller;
- LangChain-style adapter supporting think/tool/delegate/stop;
- contextual-bandit / LinUCB-style learned action scorer;
- mixed-task routing benchmark;
- benchmark evaluator and configurable experiment runtime;
- router SFT data conversion scripts and LLaMA-Factory config;
- tau2 decision-point sample assets and audit outputs.

Not finalized yet:

- paper-ready theorem proofs;
- final train/dev/test split for formal router SFT data;
- trained LoRA/SFT checkpoint artifacts;
- full PPO/GRPO-style agentic RL training;
- final paper figures and benchmark tables.

---

## 5. Quick start

### 5.1 Environment

The core synthetic experiments use the Python standard library only. Real API experiments require an OpenAI-compatible endpoint and `requests`.

Recommended setup:

```bash
conda create -n ego-router python=3.11 -y
conda activate ego-router
pip install -r requirements.txt
```

If you use the existing local conda base environment on this machine, commands can also be run as:

```bash
conda run -n base python scripts/run_synthetic_theorem_a.py
```

### 5.2 Configure API keys for real-mode experiments

Copy the example file locally:

```bash
cp .env.example .env
```

Then fill in your own values:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
OPENAI_JUDGE_MODEL=gpt-4.1-mini
```

The real `.env` file is ignored and must not be committed.

---

## 6. Reproduce key experiments

### 6.1 Synthetic theorem-A validation

```bash
python scripts/run_synthetic_theorem_a.py
```

Expected pattern:

- DP oracle performs best;
- EGO/budget-aware thresholding is close to oracle;
- fixed-depth heuristics are worse;
- never-stop-early performs poorly when continuation is costly.

### 6.2 Synthetic sweeps

```bash
python scripts/run_synthetic_sweeps.py --episodes 200
python scripts/summarize_synthetic_sweep.py
```

Outputs:

```text
outputs/synthetic_sweep_results.json
outputs/synthetic_sweep_summary.csv
outputs/synthetic_sweep_policy_metrics.csv
```

Current sweep finding: tuned EGO closely matches the DP oracle and consistently beats fixed-depth baselines, but the scalar entropy environment is too simple to create a large gap over the best tuned fixed-threshold baseline. See `docs/17_synthetic_sweep_findings.md`.

### 6.3 Mixed-task routing benchmark

```bash
python scripts/mixed_task_routing_benchmark.py
```

This evaluates heuristic vs learned routing over mixed task families.

### 6.4 Configurable benchmark experiment

Mock mode:

```bash
python scripts/run_configurable_routing_experiment.py \
  --config configs/experiment_benchmark_v3_mock.json \
  --quiet-trajectories \
  --save-json outputs/benchmark_v3_mock_results.json
```

Real OpenAI-compatible mode requires `.env` / environment variables:

```bash
python scripts/run_configurable_routing_experiment.py \
  --config configs/experiment_benchmark_v3_real.json \
  --quiet-trajectories \
  --save-json outputs/benchmark_v3_real_results.json
```

---

## 7. Data and outputs

### 7.1 Benchmark data

Benchmark task files live in:

```text
data/realistic_mixed_task_benchmark_v*.json
```

The main current benchmark file is:

```text
data/realistic_mixed_task_benchmark_v3.json
```

### 7.2 Core outputs

Core local artifacts are included under:

```text
outputs/
```

This includes:

- benchmark result JSON files;
- synthetic sweep JSON/CSV summaries;
- tau2 manual decision-point samples;
- tau2 audit and summary files.

Temporary simulation dumps such as `outputs/_tmp_*.json` and logs are ignored.

### 7.3 Training data status

Important: generated `training/llamafactory_data/router_sft_*.jsonl` files are warm-start / format-check data unless explicitly validated. The current tau2 sample assets are state-level, action-scored samples, but the final official train/dev/test router SFT split is not frozen yet.

See:

```text
docs/15_data_status_and_paths.md
```

---

## 8. Router SFT starter pack

The repo includes a LLaMA-Factory starter pack for training a small router model:

```text
training/README_router_finetune.md
training/llamafactory_configs/qwen25_router_lora_sft.yaml
training/llamafactory_data/dataset_info.json
scripts/export_router_sft_data.py
scripts/convert_benchmark_logs_to_router_sft.py
scripts/train_router_with_llamafactory.sh
```

Example:

```bash
python scripts/export_router_sft_data.py \
  --tasks data/realistic_mixed_task_benchmark_v3.json \
  --output training/llamafactory_data/router_sft_seed.jsonl

llamafactory-cli train training/llamafactory_configs/qwen25_router_lora_sft.yaml
```

No trained LoRA checkpoint is currently included in this repository.

---

## 9. Security notes

Do not commit:

- `.env`;
- API keys;
- local Claude settings;
- private benchmark dumps not intended for release;
- generated logs that may contain request payloads or credentials;
- large model checkpoints unless explicitly intended.

The repository has been cleaned to avoid committing real API keys. Use `.env.example` only as a template.

---

## 10. Suggested next steps

1. Build a mixed-regime synthetic environment where episode-level cost, budget, or entropy dynamics vary.
2. Freeze tau2 decision-point train/dev/test splits.
3. Train and evaluate a router SFT baseline.
4. Add paper figures for cost-performance, oracle gap, stopping frequency, and action distribution.
5. Extend learned routing toward offline RL or GRPO only after the SFT baseline is stable.
