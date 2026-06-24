from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.ego_core import EGOControllerConfig
from src.integrations.langchain_ego_adapter import ExpertSpec, LangChainEGOAgent, ToolSpec


class MockLLM:
    def invoke(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "delegated expert: math" in prompt_lower:
            return "Math-expert-supported answer: the derivation is valid and the final result follows rigorously."
        if "delegated expert: research" in prompt_lower:
            return "Research-expert-supported answer: the methodological framing is strong but less numerically specific."
        if "tool used: search" in prompt_lower:
            return "Search-supported answer: external evidence adds factual support."
        if "refining an answer under budget-aware orchestration" in prompt_lower:
            return "Refined answer: improved structure but still uncertain on the derivation."
        return "Initial draft: possible solution sketch, but proof details are missing."


class MockSearchTool:
    name = "search"
    description = "Retrieve external evidence for factual or latest-information queries."

    def invoke(self, query: str) -> str:
        return f"Mock factual evidence for: {query}"


class MockMathExpert:
    name = "math"
    description = "Specialist expert for derivations, equations, and proof-style reasoning."

    def invoke(self, query: str) -> str:
        return f"Math expert analysis for: {query}"


class MockResearchExpert:
    name = "research"
    description = "Specialist expert for method framing, novelty, and theorem positioning."

    def invoke(self, query: str) -> str:
        return f"Research expert analysis for: {query}"


def mock_verifier(query: str, answer: str) -> float:
    lowered = answer.lower()
    if "math-expert-supported" in lowered:
        return 3.0
    if "research-expert-supported" in lowered:
        return 2.1
    if "search-supported" in lowered:
        return 1.7
    if "refined answer" in lowered:
        return 0.6
    return -0.8


def main():
    agent = LangChainEGOAgent(
        llm=MockLLM(),
        tools=[ToolSpec(name="search", tool=MockSearchTool(), action_cost=0.16, prior_relevance=0.5)],
        experts=[
            ExpertSpec(name="math", expert=MockMathExpert(), action_cost=0.11, prior_relevance=0.8),
            ExpertSpec(name="research", expert=MockResearchExpert(), action_cost=0.13, prior_relevance=0.65),
        ],
        verifier_fn=mock_verifier,
        controller_config=EGOControllerConfig(h0=0.08, alpha_h=0.75),
        max_steps=3,
    )

    query = "Derive the result carefully and explain the proof idea for the method."
    result = agent.invoke(query)

    print("=== LangChain-Compatible Delegate EGO Example ===")
    print(f"Query: {query}")
    print(f"Final answer: {result.final_answer}")
    print(f"Stop reason: {result.decision.reason}")
    print("Trajectory:")
    for step in result.trajectory:
        print(f"- step={step['step']} chosen={step['chosen_action']} budget_before={step['budget_before']}")
        for scored in step['action_scores']:
            print(
                f"    * {scored.action_name:18s} score={scored.score:.3f} "
                f"gain={scored.estimated_gain:.3f} cost={scored.action_cost:.3f}"
            )


if __name__ == "__main__":
    main()
