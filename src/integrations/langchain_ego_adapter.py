from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.integrations.ego_core import (
    CandidatePosteriorEstimator,
    EGOActionMetrics,
    EGOBudget,
    EGOControllerConfig,
    EGODecision,
    EGOStoppingController,
)
from src.integrations.learned_action_scorer import (
    EGOFeatureBuilder,
    LinUCBActionScorer,
    ScoredAction,
)

VerifierFn = Callable[[str, str], float]
ToolRelevanceFn = Callable[[str, str, str], float]
ExpertRelevanceFn = Callable[[str, str, str], float]


@dataclass
class ToolSpec:
    name: str
    tool: Any
    description: str = ""
    action_cost: float = 0.15
    prior_relevance: float = 0.5

    def invoke(self, tool_input: str) -> Any:
        if hasattr(self.tool, "invoke"):
            return self.tool.invoke(tool_input)
        if callable(self.tool):
            return self.tool(tool_input)
        raise TypeError(f"Tool '{self.name}' is not callable and has no invoke() method.")


@dataclass
class ExpertSpec:
    name: str
    expert: Any
    description: str = ""
    action_cost: float = 0.12
    prior_relevance: float = 0.5

    def invoke(self, expert_input: str) -> Any:
        if hasattr(self.expert, "invoke"):
            return self.expert.invoke(expert_input)
        if callable(self.expert):
            return self.expert(expert_input)
        raise TypeError(f"Expert '{self.name}' is not callable and has no invoke() method.")


@dataclass
class ActionScore:
    action_name: str
    score: float
    estimated_gain: float
    action_cost: float
    rationale: str
    predicted_reward: Optional[float] = None
    exploration_bonus: Optional[float] = None
    feature_vector: Optional[List[float]] = None
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class LangChainEGOResult:
    query: str
    final_answer: str
    decision: EGODecision
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)
    verifier_scores: Dict[str, float] = field(default_factory=dict)


class LangChainEGOAgent:
    """LangChain-compatible orchestration wrapper with action scoring.

    It works with objects that behave like LangChain runnables/tools/experts but does not
    require LangChain imports at import time. Expected interfaces:
    - llm.invoke(prompt) -> str-like output
    - tools: each has .name and .invoke(input) or is callable
    - experts: each has .name and .invoke(input) or is callable

    The control policy follows the EGO budget-aware stopping rule and performs
    action scoring over THINK, TOOL(name), and DELEGATE(name) actions.
    """

    def __init__(
        self,
        llm: Any,
        tools: Optional[Sequence[Any]] = None,
        experts: Optional[Sequence[Any]] = None,
        verifier_fn: Optional[VerifierFn] = None,
        tool_relevance_fn: Optional[ToolRelevanceFn] = None,
        expert_relevance_fn: Optional[ExpertRelevanceFn] = None,
        controller_config: Optional[EGOControllerConfig] = None,
        max_steps: int = 4,
        think_action_cost: float = 0.05,
        use_learned_action_scorer: bool = False,
        learned_scorer: Optional[LinUCBActionScorer] = None,
        learned_gain_mix: float = 0.5,
    ):
        self.llm = llm
        self.verifier_fn = verifier_fn or self._default_verifier
        self.tool_relevance_fn = tool_relevance_fn or self._default_tool_relevance
        self.expert_relevance_fn = expert_relevance_fn or self._default_expert_relevance
        self.controller = EGOStoppingController(controller_config)
        self.posterior_estimator = CandidatePosteriorEstimator()
        self.max_steps = max_steps
        self.think_action_cost = think_action_cost
        self.tools = [self._normalize_tool(tool) for tool in (tools or [])]
        self.experts = [self._normalize_expert(expert) for expert in (experts or [])]
        self.use_learned_action_scorer = use_learned_action_scorer
        self.feature_builder = EGOFeatureBuilder()
        self.learned_scorer = learned_scorer or LinUCBActionScorer(feature_dim=15)
        self.learned_gain_mix = learned_gain_mix

    def invoke(self, query: str) -> LangChainEGOResult:
        budget = EGOBudget(steps_remaining=self.max_steps)
        candidates: List[str] = []
        verifier_scores: Dict[str, float] = {}
        trajectory: List[Dict[str, Any]] = []

        draft = self._call_llm(self._draft_prompt(query))
        candidates.append(draft)
        verifier_scores[draft] = self.verifier_fn(query, draft)

        while True:
            metrics = self.posterior_estimator.estimate(candidates, verifier_scores)
            decision = self.controller.decide(metrics, budget)
            action_scores = self._score_actions(query, budget, metrics)
            best_action = action_scores[0] if action_scores else None

            if decision.should_stop or best_action is None or best_action.score <= 0.0:
                final_answer = self._best_candidate(candidates, verifier_scores)
                stop_reason = decision.reason
                if best_action is None:
                    stop_reason = "no_available_actions"
                elif best_action.score <= 0.0 and not decision.should_stop:
                    stop_reason = "all_action_scores_non_positive"
                return LangChainEGOResult(
                    query=query,
                    final_answer=final_answer,
                    decision=EGODecision(
                        should_stop=True,
                        threshold=decision.threshold,
                        score=best_action.score if best_action else decision.score,
                        reason=stop_reason,
                    ),
                    trajectory=trajectory,
                    candidates=candidates,
                    verifier_scores=verifier_scores,
                )

            action_record: Dict[str, Any] = {
                "step": self.max_steps - budget.steps_remaining + 1,
                "budget_before": budget.steps_remaining,
                "metrics": metrics,
                "decision": decision,
                "action_scores": action_scores,
                "chosen_action": best_action.action_name,
            }

            next_candidate = self._execute_action(query, best_action, candidates, action_record)
            candidates.append(next_candidate)
            next_verifier = self.verifier_fn(query, next_candidate)
            verifier_scores[next_candidate] = next_verifier
            if self.use_learned_action_scorer and best_action.feature_vector is not None:
                reward_signal = max(next_verifier - metrics.verifier_confidence, 0.0)
                reward_signal += 0.25 * max(metrics.entropy - self.posterior_estimator.estimate(candidates, verifier_scores).entropy, 0.0)
                self.learned_scorer.update(best_action.action_name, best_action.feature_vector, reward_signal)
                action_record["bandit_reward"] = reward_signal
            trajectory.append(action_record)
            budget.steps_remaining -= 1

    def _score_actions(
        self,
        query: str,
        budget: EGOBudget,
        metrics: EGOActionMetrics,
    ) -> List[ActionScore]:
        if budget.steps_remaining <= 0:
            return []

        action_scores: List[ActionScore] = []

        think_gain = self._estimate_think_gain(query, budget, metrics)
        action_scores.append(
            self._make_action_score(
                action_name="think",
                metrics=metrics,
                budget=budget,
                action_cost=self.think_action_cost,
                estimated_gain=think_gain,
                rationale="cheap internal refinement",
                relevance=0.0,
                prior_relevance=0.0,
                extra_metadata={"entropy": metrics.entropy, "disagreement": metrics.disagreement},
            )
        )

        for tool in self.tools:
            tool_gain = self._estimate_tool_gain(query, tool, budget, metrics)
            relevance = self.tool_relevance_fn(query, tool.name, tool.description)
            action_scores.append(
                self._make_action_score(
                    action_name=f"tool:{tool.name}",
                    metrics=metrics,
                    budget=budget,
                    action_cost=tool.action_cost,
                    estimated_gain=tool_gain,
                    rationale="external information gathering",
                    relevance=relevance,
                    prior_relevance=tool.prior_relevance,
                    extra_metadata={
                        "tool_prior_relevance": tool.prior_relevance,
                        "entropy": metrics.entropy,
                        "disagreement": metrics.disagreement,
                    },
                )
            )

        for expert in self.experts:
            delegate_gain = self._estimate_delegate_gain(query, expert, budget, metrics)
            relevance = self.expert_relevance_fn(query, expert.name, expert.description)
            action_scores.append(
                self._make_action_score(
                    action_name=f"delegate:{expert.name}",
                    metrics=metrics,
                    budget=budget,
                    action_cost=expert.action_cost,
                    estimated_gain=delegate_gain,
                    rationale="specialized expert delegation",
                    relevance=relevance,
                    prior_relevance=expert.prior_relevance,
                    extra_metadata={
                        "expert_prior_relevance": expert.prior_relevance,
                        "entropy": metrics.entropy,
                        "disagreement": metrics.disagreement,
                    },
                )
            )

        action_scores.sort(key=lambda item: item.score, reverse=True)
        return action_scores

    def _make_action_score(
        self,
        action_name: str,
        metrics: EGOActionMetrics,
        budget: EGOBudget,
        action_cost: float,
        estimated_gain: float,
        rationale: str,
        relevance: float,
        prior_relevance: float,
        extra_metadata: Dict[str, float],
    ) -> ActionScore:
        feature = self.feature_builder.build(
            action_name=action_name,
            metrics=metrics,
            budget=budget,
            action_cost=action_cost,
            relevance=relevance,
            prior_relevance=prior_relevance,
        )
        if self.use_learned_action_scorer:
            scored: ScoredAction = self.learned_scorer.score(
                action_name=action_name,
                feature_vector=feature.values,
                action_cost=action_cost,
                metadata=extra_metadata,
            )
            hybrid_score = (
                scored.predicted_reward
                + scored.exploration_bonus
                + self.learned_gain_mix * estimated_gain
                - action_cost
            )
            return ActionScore(
                action_name=action_name,
                score=hybrid_score,
                estimated_gain=estimated_gain,
                action_cost=action_cost,
                rationale=rationale,
                predicted_reward=scored.predicted_reward,
                exploration_bonus=scored.exploration_bonus,
                feature_vector=scored.feature_vector,
                metadata={**extra_metadata, **feature.metadata},
            )
        return ActionScore(
            action_name=action_name,
            score=estimated_gain - action_cost,
            estimated_gain=estimated_gain,
            action_cost=action_cost,
            rationale=rationale,
            feature_vector=feature.values,
            metadata={**extra_metadata, **feature.metadata},
        )

    def _estimate_think_gain(
        self,
        query: str,
        budget: EGOBudget,
        metrics: EGOActionMetrics,
    ) -> float:
        budget_bonus = 0.05 * min(budget.steps_remaining, 3)
        return (
            0.55 * metrics.entropy
            + 0.20 * metrics.disagreement
            + 0.10 * (1.0 - metrics.verifier_confidence)
            + budget_bonus
        )

    def _estimate_tool_gain(
        self,
        query: str,
        tool: ToolSpec,
        budget: EGOBudget,
        metrics: EGOActionMetrics,
    ) -> float:
        relevance = self.tool_relevance_fn(query, tool.name, tool.description)
        urgency = 0.10 if budget.steps_remaining > 1 else -0.10
        return (
            0.65 * metrics.entropy
            + 0.25 * metrics.disagreement
            + 0.20 * (1.0 - metrics.verifier_confidence)
            + 0.50 * relevance
            + 0.25 * tool.prior_relevance
            + urgency
        )

    def _estimate_delegate_gain(
        self,
        query: str,
        expert: ExpertSpec,
        budget: EGOBudget,
        metrics: EGOActionMetrics,
    ) -> float:
        relevance = self.expert_relevance_fn(query, expert.name, expert.description)
        specialization_bonus = 0.20 if budget.steps_remaining > 1 else 0.0
        return (
            0.60 * metrics.entropy
            + 0.30 * metrics.disagreement
            + 0.25 * (1.0 - metrics.verifier_confidence)
            + 0.55 * relevance
            + 0.30 * expert.prior_relevance
            + specialization_bonus
        )

    def _execute_action(
        self,
        query: str,
        best_action: ActionScore,
        candidates: List[str],
        action_record: Dict[str, Any],
    ) -> str:
        if best_action.action_name == "think":
            action_record["action"] = "think"
            action_record["action_cost"] = self.think_action_cost
            return self._call_llm(self._improve_prompt(query, candidates))

        if best_action.action_name.startswith("tool:"):
            tool_name = best_action.action_name.split(":", 1)[1]
            tool = next(tool for tool in self.tools if tool.name == tool_name)
            tool_output = tool.invoke(query)
            action_record["action"] = best_action.action_name
            action_record["action_cost"] = tool.action_cost
            action_record["tool_output"] = tool_output
            return self._call_llm(
                self._tool_augmented_prompt(query, tool.name, str(tool_output), candidates)
            )

        if best_action.action_name.startswith("delegate:"):
            expert_name = best_action.action_name.split(":", 1)[1]
            expert = next(expert for expert in self.experts if expert.name == expert_name)
            expert_output = expert.invoke(query)
            action_record["action"] = best_action.action_name
            action_record["action_cost"] = expert.action_cost
            action_record["expert_output"] = expert_output
            return self._call_llm(
                self._delegate_augmented_prompt(query, expert.name, str(expert_output), candidates)
            )

        raise ValueError(f"Unknown action: {best_action.action_name}")

    def _normalize_tool(self, tool: Any) -> ToolSpec:
        if isinstance(tool, ToolSpec):
            return tool
        name = getattr(tool, "name", tool.__class__.__name__)
        description = getattr(tool, "description", "")
        return ToolSpec(name=name, tool=tool, description=description)

    def _normalize_expert(self, expert: Any) -> ExpertSpec:
        if isinstance(expert, ExpertSpec):
            return expert
        name = getattr(expert, "name", expert.__class__.__name__)
        description = getattr(expert, "description", "")
        return ExpertSpec(name=name, expert=expert, description=description)

    def _call_llm(self, prompt: str) -> str:
        if hasattr(self.llm, "invoke"):
            result = self.llm.invoke(prompt)
        elif callable(self.llm):
            result = self.llm(prompt)
        else:
            raise TypeError("LLM must be callable or expose invoke().")
        return self._coerce_to_text(result)

    def _coerce_to_text(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if hasattr(result, "content"):
            content = getattr(result, "content")
            if isinstance(content, str):
                return content
        return str(result)

    def _best_candidate(self, candidates: List[str], verifier_scores: Dict[str, float]) -> str:
        return max(candidates, key=lambda c: verifier_scores.get(c, float("-inf")))

    def _draft_prompt(self, query: str) -> str:
        return (
            "You are solving a task under EGO orchestration. "
            "Provide your current best concise answer.\n\n"
            f"Task: {query}"
        )

    def _improve_prompt(self, query: str, candidates: List[str]) -> str:
        joined = "\n---\n".join(candidates[-3:])
        return (
            "You are refining an answer under budget-aware orchestration. "
            "Given previous candidate answers, produce an improved best answer.\n\n"
            f"Task: {query}\n\n"
            f"Previous candidates:\n{joined}"
        )

    def _tool_augmented_prompt(
        self,
        query: str,
        tool_name: str,
        tool_output: str,
        candidates: List[str],
    ) -> str:
        joined = "\n---\n".join(candidates[-2:])
        return (
            "You are refining an answer with external evidence under EGO orchestration.\n\n"
            f"Task: {query}\n"
            f"Tool used: {tool_name}\n"
            f"Tool output: {tool_output}\n\n"
            f"Previous candidates:\n{joined}\n\n"
            "Return an updated best answer."
        )

    def _delegate_augmented_prompt(
        self,
        query: str,
        expert_name: str,
        expert_output: str,
        candidates: List[str],
    ) -> str:
        joined = "\n---\n".join(candidates[-2:])
        return (
            "You are refining an answer using a delegated expert under EGO orchestration.\n\n"
            f"Task: {query}\n"
            f"Delegated expert: {expert_name}\n"
            f"Expert output: {expert_output}\n\n"
            f"Previous candidates:\n{joined}\n\n"
            "Return an updated best answer."
        )

    def _default_verifier(self, query: str, answer: str) -> float:
        lowered_query = set(query.lower().split())
        lowered_answer = set(answer.lower().split())
        overlap = len(lowered_query.intersection(lowered_answer))
        return overlap / max(len(lowered_query), 1)

    def _default_tool_relevance(self, query: str, tool_name: str, description: str) -> float:
        q = query.lower()
        t = tool_name.lower()
        d = description.lower()
        score = 0.0
        keyword_map = {
            "search": ["search", "find", "latest", "look up", "retrieve", "evidence"],
            "calculator": ["calculate", "math", "equation", "compute", "number"],
            "code": ["code", "python", "program", "script", "implement"],
        }
        for key, words in keyword_map.items():
            if key in t or key in d:
                for word in words:
                    if word in q:
                        score += 0.2
        return min(score, 1.0)

    def _default_expert_relevance(self, query: str, expert_name: str, description: str) -> float:
        q = query.lower()
        e = expert_name.lower()
        d = description.lower()
        score = 0.0
        keyword_map = {
            "math": ["math", "equation", "proof", "derive", "theorem"],
            "code": ["code", "python", "implement", "script", "debug"],
            "research": ["paper", "research", "method", "novelty", "positioning"],
        }
        for key, words in keyword_map.items():
            if key in e or key in d:
                for word in words:
                    if word in q:
                        score += 0.2
        return min(score, 1.0)
