from __future__ import annotations

import argparse
import copy
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.ego_core import EGODecision
from src.integrations.langchain_ego_adapter import LangChainEGOAgent, LangChainEGOResult
from scripts.run_configurable_routing_experiment import (
    build_benchmark_evaluator,
    build_mock_agent,
    build_real_agent,
    evaluate_agent,
    load_json,
    load_tasks,
    load_dotenv,
    maybe_save_json,
    print_metrics,
    resolve_path,
    setup_logger,
)


class FixedWorkflowAgent:
    """Simple fixed-policy baselines for comparison against EGO."""

    def __init__(self, backend: LangChainEGOAgent, policy: str, max_steps: int = 3):
        self.backend = backend
        self.policy = policy
        self.max_steps = max_steps

    def invoke(self, query: str) -> LangChainEGOResult:
        candidates: List[str] = []
        verifier_scores: Dict[str, float] = {}
        trajectory: List[Dict[str, Any]] = []

        draft = self.backend._call_llm(self.backend._draft_prompt(query))
        candidates.append(draft)
        verifier_scores[draft] = self.backend.verifier_fn(query, draft)

        remaining = self.max_steps
        step_index = 1

        if self.policy == "always_think":
            while remaining > 0:
                next_candidate = self.backend._call_llm(self.backend._improve_prompt(query, candidates))
                candidates.append(next_candidate)
                verifier_scores[next_candidate] = self.backend.verifier_fn(query, next_candidate)
                trajectory.append(
                    {
                        "step": step_index,
                        "budget_before": remaining,
                        "chosen_action": "think",
                        "action": "think",
                        "action_scores": [],
                    }
                )
                remaining -= 1
                step_index += 1

        elif self.policy == "tool_first":
            if self.backend.tools and remaining > 0:
                tool = self._best_tool(query)
                tool_output = tool.invoke(query)
                next_candidate = self.backend._call_llm(
                    self.backend._tool_augmented_prompt(query, tool.name, str(tool_output), candidates)
                )
                candidates.append(next_candidate)
                verifier_scores[next_candidate] = self.backend.verifier_fn(query, next_candidate)
                trajectory.append(
                    {
                        "step": step_index,
                        "budget_before": remaining,
                        "chosen_action": f"tool:{tool.name}",
                        "action": f"tool:{tool.name}",
                        "tool_output": str(tool_output),
                        "action_scores": [],
                    }
                )
                remaining -= 1
                step_index += 1
            while remaining > 0:
                next_candidate = self.backend._call_llm(self.backend._improve_prompt(query, candidates))
                candidates.append(next_candidate)
                verifier_scores[next_candidate] = self.backend.verifier_fn(query, next_candidate)
                trajectory.append(
                    {
                        "step": step_index,
                        "budget_before": remaining,
                        "chosen_action": "think",
                        "action": "think",
                        "action_scores": [],
                    }
                )
                remaining -= 1
                step_index += 1

        elif self.policy == "delegate_first":
            if self.backend.experts and remaining > 0:
                expert = self._best_expert(query)
                expert_output = expert.invoke(query)
                next_candidate = self.backend._call_llm(
                    self.backend._delegate_augmented_prompt(query, expert.name, str(expert_output), candidates)
                )
                candidates.append(next_candidate)
                verifier_scores[next_candidate] = self.backend.verifier_fn(query, next_candidate)
                trajectory.append(
                    {
                        "step": step_index,
                        "budget_before": remaining,
                        "chosen_action": f"delegate:{expert.name}",
                        "action": f"delegate:{expert.name}",
                        "expert_output": str(expert_output),
                        "action_scores": [],
                    }
                )
                remaining -= 1
                step_index += 1
            while remaining > 0:
                next_candidate = self.backend._call_llm(self.backend._improve_prompt(query, candidates))
                candidates.append(next_candidate)
                verifier_scores[next_candidate] = self.backend.verifier_fn(query, next_candidate)
                trajectory.append(
                    {
                        "step": step_index,
                        "budget_before": remaining,
                        "chosen_action": "think",
                        "action": "think",
                        "action_scores": [],
                    }
                )
                remaining -= 1
                step_index += 1
        else:
            raise ValueError(f"Unsupported fixed policy: {self.policy}")

        final_answer = self.backend._best_candidate(candidates, verifier_scores)
        return LangChainEGOResult(
            query=query,
            final_answer=final_answer,
            decision=EGODecision(
                should_stop=True,
                threshold=0.0,
                score=0.0,
                reason=f"fixed_policy_{self.policy}_exhausted",
            ),
            trajectory=trajectory,
            candidates=candidates,
            verifier_scores=verifier_scores,
        )

    def _best_tool(self, query: str):
        scored = []
        for tool in self.backend.tools:
            relevance = self.backend.tool_relevance_fn(query, tool.name, tool.description)
            scored.append((relevance + tool.prior_relevance, tool))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _best_expert(self, query: str):
        scored = []
        for expert in self.backend.experts:
            relevance = self.backend.expert_relevance_fn(query, expert.name, expert.description)
            scored.append((relevance + expert.prior_relevance, expert))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline matrix over the benchmark.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to baseline-matrix config JSON, relative to project root or absolute.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save matrix results as JSON.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to save run logs.",
    )
    parser.add_argument(
        "--quiet-trajectories",
        action="store_true",
        help="Suppress per-query trajectories in the printed output.",
    )
    return parser.parse_args()


def load_matrix_config(path: Path) -> Dict[str, Any]:
    return load_json(path)


def build_agent_for_baseline(
    baseline_name: str,
    base_config: Dict[str, Any],
    tasks,
    mode: str,
):
    cfg = copy.deepcopy(base_config)
    judge_model = None
    if baseline_name == "heuristic_ego":
        cfg["use_learned_action_scorer"] = False
        if mode == "mock":
            agent, _ = build_mock_agent(cfg)
        else:
            agent, _, judge_model = build_real_agent(cfg, tasks)
        return agent, judge_model

    if baseline_name == "learned_ego":
        cfg["use_learned_action_scorer"] = True
        if mode == "mock":
            agent, _ = build_mock_agent(cfg)
        else:
            agent, _, judge_model = build_real_agent(cfg, tasks)
        return agent, judge_model

    if mode == "mock":
        backend, _ = build_mock_agent(cfg)
    else:
        backend, _, judge_model = build_real_agent(cfg, tasks)

    if baseline_name == "always_think":
        return FixedWorkflowAgent(backend=backend, policy="always_think", max_steps=int(cfg.get("max_steps", 3))), judge_model
    if baseline_name == "tool_first":
        return FixedWorkflowAgent(backend=backend, policy="tool_first", max_steps=int(cfg.get("max_steps", 3))), judge_model
    if baseline_name == "delegate_first":
        return FixedWorkflowAgent(backend=backend, policy="delegate_first", max_steps=int(cfg.get("max_steps", 3))), judge_model

    raise ValueError(f"Unsupported baseline: {baseline_name}")


def summarize_matrix(matrix_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for baseline_name, metrics in matrix_results.items():
        rows.append(
            {
                "baseline": baseline_name,
                "avg_reward": metrics["avg_reward"],
                "std_reward": metrics["std_reward"],
                "first_action_accuracy": metrics.get("first_action_accuracy"),
                "num_tasks": metrics["num_tasks"],
            }
        )
    rows.sort(key=lambda row: row["avg_reward"], reverse=True)
    return rows


def print_matrix_summary(rows: Sequence[Dict[str, Any]]) -> None:
    print("=== Baseline Matrix Summary ===")
    for row in rows:
        faa = row["first_action_accuracy"]
        faa_text = "n/a" if faa is None else f"{faa:.3f}"
        print(
            f"- {row['baseline']:16s} avg_reward={row['avg_reward']:.3f} "
            f"std={row['std_reward']:.3f} first_action_acc={faa_text} num_tasks={row['num_tasks']}"
        )
    print()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    logger = setup_logger(args.log_file)

    matrix_config_path = resolve_path(args.config)
    matrix_config = load_matrix_config(matrix_config_path)
    experiment_config_path = resolve_path(matrix_config["experiment_config"])
    experiment_config = load_json(experiment_config_path)
    tasks = load_tasks(resolve_path(experiment_config["query_set"]))
    baselines = list(matrix_config.get("baselines") or ["always_think", "tool_first", "delegate_first", "heuristic_ego", "learned_ego"])
    mode = experiment_config.get("mode", "mock")

    logger.info("BASELINE MATRIX loaded mode=%s num_tasks=%s baselines=%s", mode, len(tasks), baselines)

    matrix_results: Dict[str, Dict[str, Any]] = {}
    shared_judge_model = None

    for baseline_name in baselines:
        start = time.time()
        logger.info("BASELINE START name=%s", baseline_name)
        agent, judge_model = build_agent_for_baseline(baseline_name, experiment_config, tasks, mode)
        if judge_model is not None and shared_judge_model is None:
            shared_judge_model = judge_model
        evaluator = build_benchmark_evaluator(experiment_config, mode=mode, judge_model=judge_model or shared_judge_model)
        metrics = evaluate_agent(agent, tasks, evaluator, logger=logger)
        matrix_results[baseline_name] = metrics
        elapsed = time.time() - start
        logger.info(
            "BASELINE END name=%s avg_reward=%.3f first_action_accuracy=%s elapsed_sec=%.2f",
            baseline_name,
            metrics["avg_reward"],
            metrics.get("first_action_accuracy"),
            elapsed,
        )
        print(f"\n### Baseline: {baseline_name} ###")
        print_metrics(metrics, mode=mode, quiet_trajectories=args.quiet_trajectories)

    summary_rows = summarize_matrix(matrix_results)
    print_matrix_summary(summary_rows)
    payload = {
        "baseline_matrix_config": str(matrix_config_path),
        "experiment_config": str(experiment_config_path),
        "mode": mode,
        "summary_rows": summary_rows,
        "results": matrix_results,
    }
    maybe_save_json(args.save_json, payload)


if __name__ == "__main__":
    main()
