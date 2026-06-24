from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.ego_core import EGOControllerConfig
from src.integrations.langchain_ego_adapter import LangChainEGOAgent, ToolSpec


class MockLLM:
    def invoke(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "tool used: calculator" in prompt_lower:
            return "Calculator-supported answer: the numerical result is 42 with verified arithmetic."
        if "tool used: search" in prompt_lower:
            return "Search-supported answer: retrieved evidence supports the factual claim."
        if "refining an answer under budget-aware orchestration" in prompt_lower:
            return "Refined reasoning answer: still somewhat uncertain but more structured."
        return "Initial draft answer: uncertain, possibly incomplete."


class MockSearchTool:
    name = "search"
    description = "Retrieve external evidence for factual or latest-information queries."

    def invoke(self, query: str) -> str:
        return f"Mock factual evidence for: {query}"


class MockCalculatorTool:
    name = "calculator"
    description = "Compute arithmetic or symbolic expressions."

    def invoke(self, query: str) -> str:
        return f"Mock calculation result for: {query} => 42"


def mock_verifier(query: str, answer: str) -> float:
    lowered = answer.lower()
    if "calculator-supported" in lowered:
        return 2.8
    if "search-supported" in lowered:
        return 2.2
    if "refined reasoning" in lowered:
        return 0.7
    return -0.7


def main():
    agent = LangChainEGOAgent(
        llm=MockLLM(),
        tools=[
            ToolSpec(name="search", tool=MockSearchTool(), action_cost=0.16, prior_relevance=0.6),
            ToolSpec(name="calculator", tool=MockCalculatorTool(), action_cost=0.12, prior_relevance=0.7),
        ],
        verifier_fn=mock_verifier,
        controller_config=EGOControllerConfig(h0=0.08, alpha_h=0.75),
        max_steps=3,
    )

    query = "Calculate the answer carefully and use tools if helpful."
    result = agent.invoke(query)

    print("=== LangChain-Compatible Multi-Tool EGO Example ===")
    print(f"Query: {query}")
    print(f"Final answer: {result.final_answer}")
    print(f"Stop reason: {result.decision.reason}")
    print("Trajectory:")
    for step in result.trajectory:
        print(f"- step={step['step']} chosen={step['chosen_action']} budget_before={step['budget_before']}")
        scores = step['action_scores']
        for scored in scores:
            print(
                f"    * {scored.action_name:16s} score={scored.score:.3f} "
                f"gain={scored.estimated_gain:.3f} cost={scored.action_cost:.3f}"
            )


if __name__ == "__main__":
    main()
