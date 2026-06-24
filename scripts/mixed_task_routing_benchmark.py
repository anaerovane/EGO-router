from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.ego_core import EGOControllerConfig
from src.integrations.langchain_ego_adapter import ExpertSpec, LangChainEGOAgent, ToolSpec
from src.integrations.learned_action_scorer import LinUCBActionScorer


@dataclass(frozen=True)
class BenchmarkTask:
    query: str
    task_type: str
    best_action: str


TASKS: Sequence[BenchmarkTask] = [
    BenchmarkTask(
        query="Derive the theorem carefully and justify the proof idea.",
        task_type="math",
        best_action="delegate:math",
    ),
    BenchmarkTask(
        query="Calculate the numerical answer and verify the arithmetic.",
        task_type="calc",
        best_action="tool:calculator",
    ),
    BenchmarkTask(
        query="Find supporting evidence for the factual claim.",
        task_type="search",
        best_action="tool:search",
    ),
    BenchmarkTask(
        query="Implement the solution logic and debug the script behavior.",
        task_type="code",
        best_action="delegate:code",
    ),
    BenchmarkTask(
        query="Improve the reasoning internally before giving a concise answer.",
        task_type="think",
        best_action="think",
    ),
] * 8


class BenchmarkLLM:
    def invoke(self, prompt: str) -> str:
        lower = prompt.lower()
        if "delegated expert: math" in lower:
            return "Math-expert-supported answer: rigorous derivation with validated final result."
        if "delegated expert: code" in lower:
            return "Code-expert-supported answer: implementation details and debugging steps are correct."
        if "tool used: calculator" in lower:
            return "Calculator-supported answer: arithmetic verification confirms the numerical result."
        if "tool used: search" in lower:
            return "Search-supported answer: retrieved evidence confirms the factual claim."
        if "refining an answer under budget-aware orchestration" in lower:
            return "Refined reasoning answer: internally improved but not externally validated."
        return "Initial draft answer: uncertain and only partially supported."


class SearchTool:
    name = "search"
    description = "Retrieve evidence for factual and latest-information queries."
    def invoke(self, query: str) -> str:
        return f"search evidence for: {query}"


class CalculatorTool:
    name = "calculator"
    description = "Compute arithmetic or symbolic calculations."
    def invoke(self, query: str) -> str:
        return f"calculator result for: {query}"


class MathExpert:
    name = "math"
    description = "Expert for proofs, derivations, and theorem-style reasoning."
    def invoke(self, query: str) -> str:
        return f"math expert response for: {query}"


class CodeExpert:
    name = "code"
    description = "Expert for code, scripts, implementation, and debugging."
    def invoke(self, query: str) -> str:
        return f"code expert response for: {query}"


def verifier(query: str, answer: str) -> float:
    q = query.lower()
    a = answer.lower()
    score = -0.4

    if "deriv" in q or "proof" in q or "theorem" in q:
        if "math-expert-supported" in a:
            score = 3.0
        elif "refined reasoning" in a:
            score = 1.0
        elif "calculator-supported" in a:
            score = 0.4

    elif "calculate" in q or "numerical" in q or "arithmetic" in q:
        if "calculator-supported" in a:
            score = 3.0
        elif "math-expert-supported" in a:
            score = 2.2
        elif "refined reasoning" in a:
            score = 0.8

    elif "evidence" in q or "factual" in q or "claim" in q:
        if "search-supported" in a:
            score = 3.0
        elif "refined reasoning" in a:
            score = 0.7
        elif "math-expert-supported" in a:
            score = 0.5

    elif "implement" in q or "debug" in q or "script" in q:
        if "code-expert-supported" in a:
            score = 3.0
        elif "refined reasoning" in a:
            score = 0.9
        elif "search-supported" in a:
            score = 0.3

    elif "internally" in q or "concise answer" in q:
        if "refined reasoning" in a:
            score = 2.6
        elif "math-expert-supported" in a or "code-expert-supported" in a:
            score = 1.2
        elif "search-supported" in a or "calculator-supported" in a:
            score = 0.9

    return score


def make_heuristic_agent() -> LangChainEGOAgent:
    return LangChainEGOAgent(
        llm=BenchmarkLLM(),
        tools=[
            ToolSpec(name="search", tool=SearchTool(), action_cost=0.16, prior_relevance=0.55),
            ToolSpec(name="calculator", tool=CalculatorTool(), action_cost=0.12, prior_relevance=0.75),
        ],
        experts=[
            ExpertSpec(name="math", expert=MathExpert(), action_cost=0.10, prior_relevance=0.85),
            ExpertSpec(name="code", expert=CodeExpert(), action_cost=0.11, prior_relevance=0.72),
        ],
        verifier_fn=verifier,
        controller_config=EGOControllerConfig(h0=0.08, alpha_h=0.75),
        max_steps=3,
        use_learned_action_scorer=False,
    )


def make_learned_agent() -> LangChainEGOAgent:
    return LangChainEGOAgent(
        llm=BenchmarkLLM(),
        tools=[
            ToolSpec(name="search", tool=SearchTool(), action_cost=0.16, prior_relevance=0.55),
            ToolSpec(name="calculator", tool=CalculatorTool(), action_cost=0.12, prior_relevance=0.75),
        ],
        experts=[
            ExpertSpec(name="math", expert=MathExpert(), action_cost=0.10, prior_relevance=0.85),
            ExpertSpec(name="code", expert=CodeExpert(), action_cost=0.11, prior_relevance=0.72),
        ],
        verifier_fn=verifier,
        controller_config=EGOControllerConfig(h0=0.08, alpha_h=0.75),
        max_steps=3,
        use_learned_action_scorer=True,
        learned_scorer=LinUCBActionScorer(feature_dim=15, alpha=0.12, ridge=1.0),
        learned_gain_mix=0.85,
    )


def evaluate_agent(agent: LangChainEGOAgent, tasks: Sequence[BenchmarkTask]) -> Dict[str, object]:
    rewards: List[float] = []
    correct_first_actions = 0
    by_type: Dict[str, List[float]] = defaultdict(list)
    by_type_action_acc: Dict[str, List[float]] = defaultdict(list)

    for task in tasks:
        result = agent.invoke(task.query)
        reward = verifier(task.query, result.final_answer)
        rewards.append(reward)
        by_type[task.task_type].append(reward)
        if result.trajectory:
            first_action = result.trajectory[0]["chosen_action"]
            action_acc = 1.0 if first_action == task.best_action else 0.0
            correct_first_actions += int(action_acc)
            by_type_action_acc[task.task_type].append(action_acc)
        else:
            by_type_action_acc[task.task_type].append(0.0)

    return {
        "avg_reward": statistics.mean(rewards),
        "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "first_action_accuracy": correct_first_actions / len(tasks),
        "reward_by_type": {k: statistics.mean(v) for k, v in by_type.items()},
        "first_action_accuracy_by_type": {k: statistics.mean(v) for k, v in by_type_action_acc.items()},
    }


def print_metrics(name: str, metrics: Dict[str, object]) -> None:
    print(f"=== {name} ===")
    print(
        f"avg_reward={metrics['avg_reward']:.3f} +/- {metrics['std_reward']:.3f} | "
        f"first_action_acc={metrics['first_action_accuracy']:.3f}"
    )
    print("reward_by_type:")
    for task_type, value in sorted(metrics["reward_by_type"].items()):
        print(f"  - {task_type:8s}: {value:.3f}")
    print("first_action_accuracy_by_type:")
    for task_type, value in sorted(metrics["first_action_accuracy_by_type"].items()):
        print(f"  - {task_type:8s}: {value:.3f}")
    print()


def main() -> None:
    heuristic_agent = make_heuristic_agent()
    learned_agent = make_learned_agent()

    heuristic_metrics = evaluate_agent(heuristic_agent, TASKS)
    learned_metrics = evaluate_agent(learned_agent, TASKS)

    print("=== Mixed-Task Routing Benchmark ===")
    print(f"num_tasks={len(TASKS)}")
    print()
    print_metrics("Heuristic Routing", heuristic_metrics)
    print_metrics("Learned Routing", learned_metrics)

    delta_reward = learned_metrics["avg_reward"] - heuristic_metrics["avg_reward"]
    delta_acc = learned_metrics["first_action_accuracy"] - heuristic_metrics["first_action_accuracy"]
    print(f"Delta avg_reward (learned - heuristic): {delta_reward:.3f}")
    print(f"Delta first_action_acc (learned - heuristic): {delta_acc:.3f}")


if __name__ == "__main__":
    main()
