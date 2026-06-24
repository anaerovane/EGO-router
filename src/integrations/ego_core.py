from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class EGOBudget:
    steps_remaining: int
    token_budget: Optional[float] = None
    latency_budget: Optional[float] = None


@dataclass
class EGOActionMetrics:
    entropy: float
    margin: float
    disagreement: float
    verifier_confidence: float
    action_cost: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class EGODecision:
    should_stop: bool
    threshold: float
    score: float
    reason: str


@dataclass
class EGOControllerConfig:
    h0: float = 0.08
    alpha_h: float = 0.75
    beta_entropy: float = 1.0
    beta_margin: float = 0.25
    beta_disagreement: float = 0.5
    beta_verifier: float = 0.5
    min_positive_score_to_continue: float = 0.0


class EGOStoppingController:
    """Core budget-aware EGO stopping logic.

    This module is intentionally LangChain-agnostic. Adapters can feed it
    uncertainty statistics derived from any agent stack.
    """

    def __init__(self, config: Optional[EGOControllerConfig] = None):
        self.config = config or EGOControllerConfig()

    def entropy_threshold(self, budget: EGOBudget) -> float:
        return self.config.h0 + self.config.alpha_h / (budget.steps_remaining + 1)

    def continuation_score(self, metrics: EGOActionMetrics) -> float:
        return (
            self.config.beta_entropy * metrics.entropy
            - self.config.beta_margin * metrics.margin
            + self.config.beta_disagreement * metrics.disagreement
            - self.config.beta_verifier * metrics.verifier_confidence
            - metrics.action_cost
        )

    def decide(self, metrics: EGOActionMetrics, budget: EGOBudget) -> EGODecision:
        threshold = self.entropy_threshold(budget)
        score = self.continuation_score(metrics)
        if budget.steps_remaining <= 0:
            return EGODecision(
                should_stop=True,
                threshold=threshold,
                score=score,
                reason="step_budget_exhausted",
            )
        if metrics.entropy <= threshold and score <= self.config.min_positive_score_to_continue:
            return EGODecision(
                should_stop=True,
                threshold=threshold,
                score=score,
                reason="low_entropy_and_low_value_of_continuation",
            )
        if metrics.entropy <= threshold:
            return EGODecision(
                should_stop=True,
                threshold=threshold,
                score=score,
                reason="entropy_below_budget_threshold",
            )
        if score <= self.config.min_positive_score_to_continue:
            return EGODecision(
                should_stop=True,
                threshold=threshold,
                score=score,
                reason="non_positive_continuation_score",
            )
        return EGODecision(
            should_stop=False,
            threshold=threshold,
            score=score,
            reason="continue_for_more_information",
        )


class CandidatePosteriorEstimator:
    """Turn sampled candidate answers into entropy/margin/disagreement statistics."""

    def estimate(
        self,
        candidates: List[str],
        verifier_scores: Optional[Dict[str, float]] = None,
    ) -> EGOActionMetrics:
        if not candidates:
            return EGOActionMetrics(
                entropy=1.0,
                margin=0.0,
                disagreement=1.0,
                verifier_confidence=0.0,
            )

        counts: Dict[str, int] = {}
        for candidate in candidates:
            normalized = candidate.strip()
            counts[normalized] = counts.get(normalized, 0) + 1

        verifier_scores = verifier_scores or {}
        combined_scores: Dict[str, float] = {}
        for answer, count in counts.items():
            support_bonus = math.log(count + 1)
            verifier = verifier_scores.get(answer, 0.0)
            combined_scores[answer] = verifier + support_bonus

        probs = self._softmax(combined_scores)
        sorted_items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top1 = sorted_items[0][1]
        top2 = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
        best_answer = sorted_items[0][0]
        verifier_confidence = self._sigmoid(verifier_scores.get(best_answer, 0.0))
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs.values())
        disagreement = 1.0 - max(counts.values()) / len(candidates)
        if len(probs) == 1:
            entropy = max(entropy, 1.0 - verifier_confidence)
            disagreement = max(disagreement, 1.0 - verifier_confidence)
        return EGOActionMetrics(
            entropy=entropy,
            margin=top1 - top2,
            disagreement=disagreement,
            verifier_confidence=verifier_confidence,
            metadata={"num_candidates": float(len(candidates))},
        )

    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        max_score = max(scores.values())
        exps = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exps.values())
        return {k: v / total for k, v in exps.items()}

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))
