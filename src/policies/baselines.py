from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


class Policy:
    def act(self, obs: Dict[str, float]) -> str:
        raise NotImplementedError


@dataclass
class ImmediateStopPolicy(Policy):
    def act(self, obs: Dict[str, float]) -> str:
        return "stop"


@dataclass
class NeverStopEarlyPolicy(Policy):
    def act(self, obs: Dict[str, float]) -> str:
        return "stop" if obs["remaining_budget"] <= 0 else "continue"


@dataclass
class FixedDepthPolicy(Policy):
    continue_steps: int
    initial_budget: int

    def act(self, obs: Dict[str, float]) -> str:
        steps_used = self.initial_budget - int(obs["remaining_budget"])
        if steps_used >= self.continue_steps:
            return "stop"
        return "continue"


@dataclass
class FixedThresholdPolicy(Policy):
    threshold: float
    use_observed_entropy: bool = True

    def act(self, obs: Dict[str, float]) -> str:
        entropy = obs["observed_entropy"] if self.use_observed_entropy else obs["latent_entropy"]
        return "stop" if entropy <= self.threshold else "continue"


@dataclass
class BudgetAwareThresholdPolicy(Policy):
    h0: float
    beta: float
    use_observed_entropy: bool = True

    def threshold(self, remaining_budget: int) -> float:
        return self.h0 + self.beta / (remaining_budget + 1)

    def act(self, obs: Dict[str, float]) -> str:
        entropy = obs["observed_entropy"] if self.use_observed_entropy else obs["latent_entropy"]
        threshold = self.threshold(int(obs["remaining_budget"]))
        return "stop" if entropy <= threshold else "continue"


@dataclass
class OracleThresholdPolicy(Policy):
    thresholds: Dict[int, float]
    use_observed_entropy: bool = False

    def act(self, obs: Dict[str, float]) -> str:
        entropy = obs["observed_entropy"] if self.use_observed_entropy else obs["latent_entropy"]
        remaining_budget = int(obs["remaining_budget"])
        threshold = self.thresholds.get(remaining_budget, -1.0)
        if threshold < 0.0:
            return "continue"
        return "stop" if entropy <= threshold else "continue"
