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
        if "tool output" in prompt_lower:
            return "Tool-supported final answer with evidence-backed details."
        if "refining" in prompt_lower:
            return "Refined answer with better precision but still partial evidence."
        return "Initial draft answer with uncertainty and missing evidence."


class MockSearchTool:
    name = "search"
    description = "Returns mock external evidence."

    def invoke(self, query: str) -> str:
        return f"Mock evidence for query: {query}"


def mock_verifier(query: str, answer: str) -> float:
    lowered = answer.lower()
    if "tool-supported" in lowered or "evidence-backed" in lowered:
        return 2.5
    if "refined" in lowered:
        return 0.8
    return -0.8


def main():
    agent = LangChainEGOAgent(
        llm=MockLLM(),
        tools=[ToolSpec(name="search", tool=MockSearchTool(), action_cost=0.15)],
        verifier_fn=mock_verifier,
        controller_config=EGOControllerConfig(h0=0.08, alpha_h=0.75),
        max_steps=3,
    )

    result = agent.invoke("Search for evidence and answer the question carefully.")
    print("=== LangChain-Compatible EGO Example ===")
    print(f"Final answer: {result.final_answer}")
    print(f"Stop reason: {result.decision.reason}")
    print(f"Num candidates: {len(result.candidates)}")
    print("Trajectory:")
    for step in result.trajectory:
        print(f"- step={step['step']} action={step['action']} budget_before={step['budget_before']}")


if __name__ == "__main__":
    main()
