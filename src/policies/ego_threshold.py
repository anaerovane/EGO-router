from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class EGOEntropyGatePolicy:
    """Minimal EGO-style stopping controller for the scalar synthetic environment.

    This policy intentionally only models the entropy gate.
    Richer action routing is deferred to the next environment layer.
    """

    h0: float
    alpha_h: float
    use_observed_entropy: bool = True

    def entropy_threshold(self, remaining_budget: int) -> float:
        return self.h0 + self.alpha_h / (remaining_budget + 1)

    def act(self, obs: Dict[str, float]) -> str:
        entropy = obs["observed_entropy"] if self.use_observed_entropy else obs["latent_entropy"]
        remaining_budget = int(obs["remaining_budget"])
        threshold = self.entropy_threshold(remaining_budget)
        return "stop" if entropy <= threshold else "continue"
