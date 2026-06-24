from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import random


@dataclass(frozen=True)
class EnvConfig:
    max_entropy: float = 1.0
    budget: int = 8
    rho: float = 0.25
    gamma: float = 1.0
    process_noise: float = 0.03
    observation_noise: float = 0.0
    alpha: float = 0.8
    continuation_cost: float = 0.08
    scarcity_cost_scale: float = 0.0
    init_entropy_low: float = 0.1
    init_entropy_high: float = 0.95


class SyntheticEntropyEnv:
    """Theorem-A-aligned scalar-entropy environment.

    State:
        latent entropy H_t in [0, max_entropy]
        remaining budget b_t in {0, ..., budget}

    Actions:
        - "continue": pay continuation cost and shrink entropy stochastically
        - "stop": terminate and receive stop utility

    Utility convention:
        continuing incurs negative immediate reward = -continuation_cost
        stopping yields terminal reward = 1 - alpha * H_t
    """

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None):
        self.config = config or EnvConfig()
        self.rng = random.Random(seed)
        self.latent_entropy = 0.0
        self.remaining_budget = self.config.budget
        self.done = False

    def reset(self, initial_entropy: Optional[float] = None) -> Dict[str, float]:
        if initial_entropy is None:
            initial_entropy = self.rng.uniform(
                self.config.init_entropy_low,
                self.config.init_entropy_high,
            )
        self.latent_entropy = self._clip(initial_entropy)
        self.remaining_budget = self.config.budget
        self.done = False
        return self.observe()

    def observe(self) -> Dict[str, float]:
        observed_entropy = self.latent_entropy
        if self.config.observation_noise > 0.0:
            observed_entropy += self.rng.uniform(
                -self.config.observation_noise,
                self.config.observation_noise,
            )
        observed_entropy = self._clip(observed_entropy)
        return {
            "latent_entropy": self.latent_entropy,
            "observed_entropy": observed_entropy,
            "remaining_budget": self.remaining_budget,
        }

    def stop_utility(self, entropy: Optional[float] = None) -> float:
        entropy = self.latent_entropy if entropy is None else entropy
        return 1.0 - self.config.alpha * self._clip(entropy)

    def expected_next_entropy(self, entropy: float) -> float:
        # Noise has zero mean, so expected next entropy is deterministic here.
        reduced = entropy - self.config.rho * (max(entropy, 0.0) ** self.config.gamma)
        return self._clip(reduced)

    def transition(self, entropy: float) -> float:
        noise = 0.0
        if self.config.process_noise > 0.0:
            noise = self.rng.uniform(-self.config.process_noise, self.config.process_noise)
        next_entropy = entropy - self.config.rho * (max(entropy, 0.0) ** self.config.gamma) + noise
        return self._clip(next_entropy)

    def continuation_cost(self, remaining_budget: Optional[int] = None) -> float:
        remaining_budget = self.remaining_budget if remaining_budget is None else remaining_budget
        if self.config.scarcity_cost_scale <= 0.0:
            return self.config.continuation_cost
        return self.config.continuation_cost + self.config.scarcity_cost_scale / max(remaining_budget, 1)

    def step(self, action: str) -> Tuple[Dict[str, float], float, bool, Dict[str, float]]:
        if self.done:
            raise RuntimeError("Cannot step: environment is already done.")
        if action not in {"continue", "stop"}:
            raise ValueError(f"Unknown action: {action}")

        info: Dict[str, float] = {}

        if action == "stop" or self.remaining_budget <= 0:
            reward = self.stop_utility()
            self.done = True
            info["action"] = "stop"
            info["stop_utility"] = reward
            return self.observe(), reward, True, info

        reward = -self.continuation_cost(self.remaining_budget)
        prev_entropy = self.latent_entropy
        self.latent_entropy = self.transition(self.latent_entropy)
        self.remaining_budget -= 1
        info.update(
            {
                "action": "continue",
                "previous_entropy": prev_entropy,
                "next_entropy": self.latent_entropy,
                "entropy_reduction": max(prev_entropy - self.latent_entropy, 0.0),
            }
        )

        if self.remaining_budget <= 0:
            terminal_reward = self.stop_utility()
            reward += terminal_reward
            self.done = True
            info["auto_stop_utility"] = terminal_reward
            return self.observe(), reward, True, info

        return self.observe(), reward, False, info

    def simulate_episode(self, policy, initial_entropy: Optional[float] = None) -> Dict[str, object]:
        obs = self.reset(initial_entropy=initial_entropy)
        total_reward = 0.0
        trajectory = []
        while not self.done:
            action = policy.act(obs)
            next_obs, reward, done, info = self.step(action)
            trajectory.append(
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs,
                    "done": done,
                    "info": info,
                }
            )
            total_reward += reward
            obs = next_obs
        return {
            "total_reward": total_reward,
            "trajectory": trajectory,
            "final_observation": obs,
        }

    def _clip(self, value: float) -> float:
        return min(max(value, 0.0), self.config.max_entropy)
