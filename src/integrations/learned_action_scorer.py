from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math

from src.integrations.ego_core import EGOActionMetrics, EGOBudget


@dataclass
class ActionFeatureVector:
    action_name: str
    values: List[float]
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScoredAction:
    action_name: str
    score: float
    predicted_reward: float
    exploration_bonus: float
    action_cost: float
    feature_vector: List[float]
    metadata: Dict[str, float] = field(default_factory=dict)


class EGOFeatureBuilder:
    """Build simple linear features for think/tool/delegate actions."""

    def build(
        self,
        action_name: str,
        metrics: EGOActionMetrics,
        budget: EGOBudget,
        action_cost: float,
        relevance: float = 0.0,
        prior_relevance: float = 0.0,
    ) -> ActionFeatureVector:
        kind = self._action_kind(action_name)
        values = [
            1.0,
            metrics.entropy,
            metrics.margin,
            metrics.disagreement,
            metrics.verifier_confidence,
            float(budget.steps_remaining),
            action_cost,
            relevance,
            prior_relevance,
            1.0 if kind == "think" else 0.0,
            1.0 if kind == "tool" else 0.0,
            1.0 if kind == "delegate" else 0.0,
            metrics.entropy * relevance,
            metrics.disagreement * relevance,
            (1.0 - metrics.verifier_confidence) * relevance,
        ]
        return ActionFeatureVector(
            action_name=action_name,
            values=values,
            metadata={
                "kind_think": 1.0 if kind == "think" else 0.0,
                "kind_tool": 1.0 if kind == "tool" else 0.0,
                "kind_delegate": 1.0 if kind == "delegate" else 0.0,
            },
        )

    def _action_kind(self, action_name: str) -> str:
        if action_name == "think":
            return "think"
        if action_name.startswith("tool:"):
            return "tool"
        if action_name.startswith("delegate:"):
            return "delegate"
        return "other"


class LinUCBActionScorer:
    """A lightweight per-action linear UCB scorer.

    This is intentionally simple so it maps cleanly to a contextual-bandit story
    in the paper. Each action gets its own ridge-regression parameters.
    """

    def __init__(self, feature_dim: int, alpha: float = 0.8, ridge: float = 1.0):
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.ridge = ridge
        self.A: Dict[str, List[List[float]]] = {}
        self.b: Dict[str, List[float]] = {}

    def score(
        self,
        action_name: str,
        feature_vector: List[float],
        action_cost: float,
        metadata: Optional[Dict[str, float]] = None,
    ) -> ScoredAction:
        self._ensure_action(action_name)
        theta = self._solve_theta(self.A[action_name], self.b[action_name])
        predicted_reward = self._dot(theta, feature_vector)
        a_inv_x = self._solve_linear_system(self.A[action_name], feature_vector)
        exploration_bonus = self.alpha * math.sqrt(max(self._dot(feature_vector, a_inv_x), 0.0))
        score = predicted_reward + exploration_bonus - action_cost
        return ScoredAction(
            action_name=action_name,
            score=score,
            predicted_reward=predicted_reward,
            exploration_bonus=exploration_bonus,
            action_cost=action_cost,
            feature_vector=feature_vector,
            metadata=metadata or {},
        )

    def update(self, action_name: str, feature_vector: List[float], reward: float) -> None:
        self._ensure_action(action_name)
        A = self.A[action_name]
        b = self.b[action_name]
        for i in range(self.feature_dim):
            b[i] += reward * feature_vector[i]
            for j in range(self.feature_dim):
                A[i][j] += feature_vector[i] * feature_vector[j]

    def _ensure_action(self, action_name: str) -> None:
        if action_name in self.A:
            return
        self.A[action_name] = [
            [self.ridge if i == j else 0.0 for j in range(self.feature_dim)]
            for i in range(self.feature_dim)
        ]
        self.b[action_name] = [0.0 for _ in range(self.feature_dim)]

    def _solve_theta(self, A: List[List[float]], b: List[float]) -> List[float]:
        return self._solve_linear_system(A, b)

    def _solve_linear_system(self, A: List[List[float]], b: List[float]) -> List[float]:
        n = len(b)
        aug = [row[:] + [b_i] for row, b_i in zip(A, b)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot][col]) < 1e-10:
                continue
            aug[col], aug[pivot] = aug[pivot], aug[col]
            pivot_val = aug[col][col]
            for j in range(col, n + 1):
                aug[col][j] /= pivot_val
            for row in range(n):
                if row == col:
                    continue
                factor = aug[row][col]
                if abs(factor) < 1e-12:
                    continue
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]
        return [aug[i][n] for i in range(n)]

    def _dot(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))
