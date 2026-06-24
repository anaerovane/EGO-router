from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import json
import logging
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.benchmark_evaluator import BenchmarkEvaluator
from src.integrations.ego_core import EGOControllerConfig
from src.integrations.experiment_runtime import (
    LLMJudgeVerifier,
    LocalCorpusSearchTool,
    MockCodeExpert,
    MockKeywordLLM,
    MockMathExpert,
    MockSearchTool,
    OpenAICompatibleChatModel,
    PromptedExpert,
    QueryTask,
    SafeCalculatorTool,
    load_dotenv,
    mock_keyword_verifier,
)
from src.integrations.langchain_ego_adapter import ExpertSpec, LangChainEGOAgent, ToolSpec
from src.integrations.learned_action_scorer import LinUCBActionScorer

VerifierFn = Callable[[str, str], float]


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("benchmark_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file:
        path = resolve_path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mock or real EGO routing experiments from config.")
    parser.add_argument(
        "--config",
        default="configs/experiment_mock.json",
        help="Path to experiment config JSON, relative to project root or absolute.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save metrics/result summary as JSON.",
    )
    parser.add_argument(
        "--quiet-trajectories",
        action="store_true",
        help="Print only aggregate metrics, not per-query trajectories.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to save structured run logs.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_tasks(path: Path) -> List[QueryTask]:
    raw_tasks = load_json(path)
    return [
        QueryTask(
            id=item.get("id"),
            query=item["query"],
            task_type=item.get("task_type", "general"),
            best_action=item.get("best_action"),
            reference_answer=item.get("reference_answer"),
            reference_points=list(item.get("reference_points") or []),
            rubric=list(item.get("rubric") or []),
            evaluation_type=item.get("evaluation_type"),
            metadata=dict(item.get("metadata") or {}),
        )
        for item in raw_tasks
    ]


def make_controller_config(config: Dict[str, object]) -> EGOControllerConfig:
    controller_cfg = config.get("controller_config", {})
    return EGOControllerConfig(
        h0=controller_cfg.get("h0", 0.08),
        alpha_h=controller_cfg.get("alpha_h", 0.75),
    )


def make_learned_scorer(config: Dict[str, object]) -> LinUCBActionScorer:
    scorer_cfg = config.get("learned_scorer", {})
    return LinUCBActionScorer(
        feature_dim=int(scorer_cfg.get("feature_dim", 15)),
        alpha=float(scorer_cfg.get("alpha", 0.12)),
        ridge=float(scorer_cfg.get("ridge", 1.0)),
    )


def build_mock_agent(config: Dict[str, object]) -> tuple[LangChainEGOAgent, VerifierFn]:
    costs = config.get("costs", {})
    priors = config.get("priors", {})
    verifier = mock_keyword_verifier
    agent = LangChainEGOAgent(
        llm=MockKeywordLLM(),
        tools=[
            ToolSpec(name="search", tool=MockSearchTool(), action_cost=costs.get("search", 0.16), prior_relevance=priors.get("search", 0.55)),
            ToolSpec(name="calculator", tool=SafeCalculatorTool(), action_cost=costs.get("calculator", 0.12), prior_relevance=priors.get("calculator", 0.75)),
        ],
        experts=[
            ExpertSpec(name="math", expert=MockMathExpert(), action_cost=costs.get("math", 0.10), prior_relevance=priors.get("math", 0.85)),
            ExpertSpec(name="code", expert=MockCodeExpert(), action_cost=costs.get("code", 0.11), prior_relevance=priors.get("code", 0.72)),
        ],
        verifier_fn=verifier,
        controller_config=make_controller_config(config),
        max_steps=int(config.get("max_steps", 3)),
        think_action_cost=float(costs.get("think", 0.05)),
        use_learned_action_scorer=bool(config.get("use_learned_action_scorer", False)),
        learned_scorer=make_learned_scorer(config),
        learned_gain_mix=float(config.get("learned_gain_mix", 0.85)),
    )
    return agent, verifier


def build_real_agent(config: Dict[str, object], tasks: Sequence[QueryTask]) -> tuple[LangChainEGOAgent, VerifierFn, OpenAICompatibleChatModel]:
    real_cfg = config.get("real", {})
    costs = config.get("costs", {})
    priors = config.get("priors", {})

    api_key = os.environ.get(real_cfg.get("api_key_env", "OPENAI_API_KEY"), "")
    api_base = os.environ.get(real_cfg.get("api_base_env", "OPENAI_BASE_URL"), real_cfg.get("default_api_base", "https://api.openai.com/v1"))
    model_name = os.environ.get(real_cfg.get("model_env", "OPENAI_MODEL"), real_cfg.get("default_model", "gpt-4.1-mini"))
    judge_model_name = os.environ.get(real_cfg.get("judge_model_env", "OPENAI_JUDGE_MODEL"), real_cfg.get("judge_default_model", model_name))

    if not api_key:
        raise ValueError(
            "Missing API key for real mode. Set OPENAI_API_KEY or update the env variable names in configs/experiment_real.json."
        )

    llm = OpenAICompatibleChatModel(
        api_key=api_key,
        model=model_name,
        api_base=api_base,
        temperature=float(real_cfg.get("temperature", 0.2)),
    )
    judge_llm = OpenAICompatibleChatModel(
        api_key=api_key,
        model=judge_model_name,
        api_base=api_base,
        temperature=0.0,
    )

    tools: List[ToolSpec] = []
    if bool(real_cfg.get("enable_search_tool", True)):
        search_dir = resolve_path(real_cfg.get("search_corpus_dir", "docs"))
        tools.append(
            ToolSpec(
                name="search",
                tool=LocalCorpusSearchTool(search_dir),
                action_cost=costs.get("search", 0.16),
                prior_relevance=priors.get("search", 0.60),
            )
        )
    if bool(real_cfg.get("enable_calculator_tool", True)):
        tools.append(
            ToolSpec(
                name="calculator",
                tool=SafeCalculatorTool(),
                action_cost=costs.get("calculator", 0.12),
                prior_relevance=priors.get("calculator", 0.72),
            )
        )

    experts: List[ExpertSpec] = []
    if bool(real_cfg.get("enable_math_expert", True)):
        experts.append(
            ExpertSpec(
                name="math",
                expert=PromptedExpert(
                    llm,
                    system_prompt=(
                        "You are a math expert. Prioritize rigorous derivations, explicit assumptions, and proof-style reasoning."
                    ),
                ),
                action_cost=costs.get("math", 0.10),
                prior_relevance=priors.get("math", 0.82),
            )
        )
    if bool(real_cfg.get("enable_code_expert", True)):
        experts.append(
            ExpertSpec(
                name="code",
                expert=PromptedExpert(
                    llm,
                    system_prompt=(
                        "You are a code expert. Prioritize implementation details, module-level reasoning, debugging steps, and precise edits."
                    ),
                ),
                action_cost=costs.get("code", 0.11),
                prior_relevance=priors.get("code", 0.74),
            )
        )

    query_to_type = {task.query: task.task_type for task in tasks}
    internal_verifier = LLMJudgeVerifier(model=judge_llm, query_to_task_type=query_to_type)

    agent = LangChainEGOAgent(
        llm=llm,
        tools=tools,
        experts=experts,
        verifier_fn=internal_verifier,
        controller_config=make_controller_config(config),
        max_steps=int(config.get("max_steps", 3)),
        think_action_cost=float(costs.get("think", 0.05)),
        use_learned_action_scorer=bool(config.get("use_learned_action_scorer", False)),
        learned_scorer=make_learned_scorer(config),
        learned_gain_mix=float(config.get("learned_gain_mix", 0.65)),
    )
    return agent, internal_verifier, judge_llm


def build_benchmark_evaluator(config: Dict[str, object], mode: str, judge_model: Optional[OpenAICompatibleChatModel] = None) -> BenchmarkEvaluator:
    benchmark_cfg = config.get("benchmark", {})
    use_llm = bool(benchmark_cfg.get("use_llm_for_open_ended", mode == "real"))
    return BenchmarkEvaluator(judge_model=judge_model, use_llm_for_open_ended=use_llm)


def evaluate_agent(agent: LangChainEGOAgent, tasks: Sequence[QueryTask], evaluator: BenchmarkEvaluator, logger: Optional[logging.Logger] = None) -> Dict[str, object]:
    rewards: List[float] = []
    correct_first_actions = 0
    first_action_count = 0
    by_type: Dict[str, List[float]] = defaultdict(list)
    by_type_action_acc: Dict[str, List[float]] = defaultdict(list)
    by_difficulty: Dict[str, List[float]] = defaultdict(list)
    by_split: Dict[str, List[float]] = defaultdict(list)
    by_eval_type: Dict[str, List[float]] = defaultdict(list)
    trajectories: List[Dict[str, object]] = []

    run_start = time.time()
    for index, task in enumerate(tasks, start=1):
        task_start = time.time()
        if logger:
            logger.info(
                "START task=%s/%s id=%s type=%s difficulty=%s split=%s eval=%s",
                index, len(tasks), task.id, task.task_type, task.metadata.get("difficulty"), task.metadata.get("split"), task.evaluation_type
            )
        result = agent.invoke(task.query)
        eval_result = evaluator.evaluate(task, result.final_answer)
        reward = eval_result.score
        rewards.append(reward)
        by_type[task.task_type].append(reward)
        by_difficulty[task.metadata.get("difficulty", "unknown")].append(reward)
        by_split[task.metadata.get("split", "unknown")].append(reward)
        by_eval_type[task.evaluation_type or "unknown"].append(reward)

        first_action = None
        if result.trajectory:
            first_action = result.trajectory[0]["chosen_action"]
        if first_action is not None and task.best_action is not None:
            action_acc = 1.0 if first_action == task.best_action else 0.0
            correct_first_actions += int(action_acc)
            first_action_count += 1
            by_type_action_acc[task.task_type].append(action_acc)
        elif task.best_action is not None:
            by_type_action_acc[task.task_type].append(0.0)
            first_action_count += 1

        task_elapsed = time.time() - task_start
        if logger:
            logger.info(
                "END task=%s/%s id=%s first_action=%s reward=%.3f stop_reason=%s elapsed_sec=%.2f",
                index, len(tasks), task.id, first_action, reward, result.decision.reason, task_elapsed
            )

        trajectories.append(
            {
                "id": task.id,
                "query": task.query,
                "task_type": task.task_type,
                "difficulty": task.metadata.get("difficulty"),
                "split": task.metadata.get("split"),
                "evaluation_type": task.evaluation_type,
                "best_action": task.best_action,
                "first_action": first_action,
                "final_answer": result.final_answer,
                "stop_reason": result.decision.reason,
                "reward": reward,
                "evaluation": {
                    "matched_reference": eval_result.matched_reference,
                    "matched_points": eval_result.matched_points,
                    "total_points": eval_result.total_points,
                    "details": eval_result.details,
                },
                "trajectory": [
                    {
                        "step": step["step"],
                        "budget_before": step["budget_before"],
                        "chosen_action": step["chosen_action"],
                        "action": step.get("action"),
                        "action_scores": [
                            {
                                "action_name": scored.action_name,
                                "score": scored.score,
                                "estimated_gain": scored.estimated_gain,
                                "action_cost": scored.action_cost,
                                "predicted_reward": scored.predicted_reward,
                                "exploration_bonus": scored.exploration_bonus,
                            }
                            for scored in step["action_scores"]
                        ],
                    }
                    for step in result.trajectory
                ],
            }
        )

    total_elapsed = time.time() - run_start
    if logger:
        logger.info("RUN COMPLETE num_tasks=%s avg_reward=%.3f elapsed_sec=%.2f", len(tasks), statistics.mean(rewards) if rewards else 0.0, total_elapsed)

    metrics = {
        "avg_reward": statistics.mean(rewards) if rewards else 0.0,
        "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "first_action_accuracy": (correct_first_actions / first_action_count) if first_action_count > 0 else None,
        "reward_by_type": {k: statistics.mean(v) for k, v in by_type.items()},
        "reward_by_difficulty": {k: statistics.mean(v) for k, v in by_difficulty.items()},
        "reward_by_split": {k: statistics.mean(v) for k, v in by_split.items()},
        "reward_by_evaluation_type": {k: statistics.mean(v) for k, v in by_eval_type.items()},
        "first_action_accuracy_by_type": {k: statistics.mean(v) for k, v in by_type_action_acc.items()},
        "num_tasks": len(tasks),
        "trajectories": trajectories,
    }
    return metrics


def _print_grouped_metric(title: str, values: Dict[str, float]) -> None:
    if not values:
        return
    print(f"{title}:")
    for key, value in sorted(values.items()):
        print(f"  - {key:20s}: {value:.3f}")


def print_metrics(metrics: Dict[str, object], mode: str, quiet_trajectories: bool = False) -> None:
    print(f"=== Configurable EGO Routing Experiment ({mode}) ===")
    print(
        f"num_tasks={metrics['num_tasks']} | avg_reward={metrics['avg_reward']:.3f} +/- {metrics['std_reward']:.3f}"
    )
    first_action_accuracy = metrics.get("first_action_accuracy")
    if first_action_accuracy is not None:
        print(f"first_action_accuracy={first_action_accuracy:.3f}")
    _print_grouped_metric("reward_by_type", metrics.get("reward_by_type", {}))
    _print_grouped_metric("reward_by_difficulty", metrics.get("reward_by_difficulty", {}))
    _print_grouped_metric("reward_by_split", metrics.get("reward_by_split", {}))
    _print_grouped_metric("reward_by_evaluation_type", metrics.get("reward_by_evaluation_type", {}))
    _print_grouped_metric("first_action_accuracy_by_type", metrics.get("first_action_accuracy_by_type", {}))
    print()
    if quiet_trajectories:
        return
    for trajectory in metrics["trajectories"]:
        print(f"Query: {trajectory['query']}")
        print(
            f"  id={trajectory['id']} task_type={trajectory['task_type']} difficulty={trajectory['difficulty']} split={trajectory['split']}"
        )
        print(
            f"  eval_type={trajectory['evaluation_type']} best_action={trajectory['best_action']} first_action={trajectory['first_action']}"
        )
        print(f"  reward={trajectory['reward']:.3f} stop_reason={trajectory['stop_reason']}")
        print(f"  final_answer={trajectory['final_answer']}")
        print(f"  evaluation={trajectory['evaluation']}")
        for step in trajectory["trajectory"]:
            print(f"    - step={step['step']} chosen={step['chosen_action']} budget_before={step['budget_before']}")
        print()


def maybe_save_json(path_str: Optional[str], payload: Dict[str, object]) -> None:
    if not path_str:
        return
    path = resolve_path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON results to: {path}")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    logger = setup_logger(args.log_file)
    config_path = resolve_path(args.config)
    config = load_json(config_path)
    task_path = resolve_path(config["query_set"])
    tasks = load_tasks(task_path)

    mode = config.get("mode", "mock")
    if mode == "mock":
        agent, _internal_verifier = build_mock_agent(config)
        evaluator = build_benchmark_evaluator(config, mode=mode, judge_model=None)
    elif mode == "real":
        agent, _internal_verifier, judge_model = build_real_agent(config, tasks)
        evaluator = build_benchmark_evaluator(config, mode=mode, judge_model=judge_model)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    logger.info("CONFIG loaded mode=%s query_set=%s num_tasks=%s", mode, task_path, len(tasks))
    metrics = evaluate_agent(agent, tasks, evaluator, logger=logger)
    payload = {
        "config_path": str(config_path),
        "mode": mode,
        **metrics,
    }
    print_metrics(metrics, mode, quiet_trajectories=args.quiet_trajectories)
    maybe_save_json(args.save_json, payload)


if __name__ == "__main__":
    main()
