from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.ego_core import EGOControllerConfig
from src.integrations.langchain_ego_adapter import ExpertSpec, LangChainEGOAgent, ToolSpec
from src.integrations.learned_action_scorer import LinUCBActionScorer


class MockLLM:
    def invoke(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "delegated expert: math" in prompt_lower:
            return "Math-expert-supported answer: rigorous derivation with validated final result."
        if "delegated expert: code" in prompt_lower:
            return "Code-expert-supported answer: executable reasoning with implementation details."
        if "tool used: calculator" in prompt_lower:
            return "Calculator-supported answer: arithmetic check confirms the value 42."
        if "tool used: search" in prompt_lower:
            return "Search-supported answer: external evidence confirms the factual statement."
        if "refining an answer under budget-aware orchestration" in prompt_lower:
            return "Refined answer: better than the draft but still not fully verified."
        return "Initial draft answer with uncertainty and incomplete validation."


class MockSearchTool:
    name = "search"
    description = "Retrieve evidence for factual questions."
    def invoke(self, query: str) -> str:
        return f"search evidence for: {query}"


class MockCalculatorTool:
    name = "calculator"
    description = "Compute arithmetic or symbolic expressions."
    def invoke(self, query: str) -> str:
        return f"calculator result for: {query}"


class MockMathExpert:
    name = "math"
    description = "Expert for proofs, derivations, and equations."
    def invoke(self, query: str) -> str:
        return f"math expert response for: {query}"


class MockCodeExpert:
    name = "code"
    description = "Expert for code, scripts, and implementation details."
    def invoke(self, query: str) -> str:
        return f"code expert response for: {query}"


def verifier(query: str, answer: str) -> float:
    lowered = answer.lower()
    if "math-expert-supported" in lowered:
        return 3.2
    if "code-expert-supported" in lowered:
        return 2.5
    if "calculator-supported" in lowered:
        return 2.2
    if "search-supported" in lowered:
        return 1.9
    if "refined answer" in lowered:
        return 0.8
    return -0.7


def build_agent() -> LangChainEGOAgent:
    return LangChainEGOAgent(
        llm=MockLLM(),
        tools=[
            ToolSpec(name="search", tool=MockSearchTool(), action_cost=0.16, prior_relevance=0.55),
            ToolSpec(name="calculator", tool=MockCalculatorTool(), action_cost=0.12, prior_relevance=0.75),
        ],
        experts=[
            ExpertSpec(name="math", expert=MockMathExpert(), action_cost=0.10, prior_relevance=0.85),
            ExpertSpec(name="code", expert=MockCodeExpert(), action_cost=0.11, prior_relevance=0.70),
        ],
        verifier_fn=verifier,
        controller_config=EGOControllerConfig(h0=0.08, alpha_h=0.75),
        max_steps=3,
        use_learned_action_scorer=True,
        learned_scorer=LinUCBActionScorer(feature_dim=15, alpha=0.15, ridge=1.0),
        learned_gain_mix=0.9,
    )


def run_query(query: str) -> None:
    agent = build_agent()
    result = agent.invoke(query)
    print(f"Query: {query}")
    print(f"Final answer: {result.final_answer}")
    print(f"Stop reason: {result.decision.reason}")
    for step in result.trajectory:
        print(f"- step={step['step']} chosen={step['chosen_action']} reward={step.get('bandit_reward', 0.0):.3f}")
        for scored in step['action_scores']:
            pr = scored.predicted_reward if scored.predicted_reward is not None else 0.0
            eb = scored.exploration_bonus if scored.exploration_bonus is not None else 0.0
            print(
                f"    * {scored.action_name:18s} score={scored.score:.3f} pred={pr:.3f} bonus={eb:.3f} cost={scored.action_cost:.3f}"
            )
    print()


def main():
    print("=== Learned Action Scoring Demo ===")
    run_query("Derive the formula carefully and justify the proof.")
    run_query("Calculate the numerical answer and verify it.")
    run_query("Find supporting evidence for the factual claim.")


if __name__ == "__main__":
    main()
