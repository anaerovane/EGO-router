from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.envs.synthetic_entropy_env import EnvConfig


@dataclass
class OracleResult:
    entropy_grid: List[float]
    values: Dict[int, List[float]]
    stop_mask: Dict[int, List[bool]]
    thresholds: Dict[int, float]


class DynamicProgrammingOracle:
    def __init__(self, config: EnvConfig, grid_size: int = 201):
        self.config = config
        self.grid_size = grid_size
        self.entropy_grid = [
            i * config.max_entropy / (grid_size - 1) for i in range(grid_size)
        ]

    def stop_utility(self, entropy: float) -> float:
        return 1.0 - self.config.alpha * min(max(entropy, 0.0), self.config.max_entropy)

    def expected_next_entropy(self, entropy: float) -> float:
        reduced = entropy - self.config.rho * (max(entropy, 0.0) ** self.config.gamma)
        return min(max(reduced, 0.0), self.config.max_entropy)

    def interpolate(self, values: List[float], entropy: float) -> float:
        entropy = min(max(entropy, 0.0), self.config.max_entropy)
        if entropy <= self.entropy_grid[0]:
            return values[0]
        if entropy >= self.entropy_grid[-1]:
            return values[-1]
        step = self.config.max_entropy / (self.grid_size - 1)
        left_idx = int(entropy / step)
        right_idx = min(left_idx + 1, self.grid_size - 1)
        left = self.entropy_grid[left_idx]
        right = self.entropy_grid[right_idx]
        if right == left:
            return values[left_idx]
        weight = (entropy - left) / (right - left)
        return (1 - weight) * values[left_idx] + weight * values[right_idx]

    def continuation_cost(self, budget: int) -> float:
        if self.config.scarcity_cost_scale <= 0.0:
            return self.config.continuation_cost
        return self.config.continuation_cost + self.config.scarcity_cost_scale / max(budget, 1)

    def solve(self) -> OracleResult:
        values: Dict[int, List[float]] = {}
        stop_mask: Dict[int, List[bool]] = {}
        thresholds: Dict[int, float] = {}

        values[0] = [self.stop_utility(h) for h in self.entropy_grid]
        stop_mask[0] = [True for _ in self.entropy_grid]
        thresholds[0] = self.config.max_entropy

        for budget in range(1, self.config.budget + 1):
            current_values: List[float] = []
            current_stop_mask: List[bool] = []
            threshold = None
            prev_values = values[budget - 1]
            for entropy in self.entropy_grid:
                stop_value = self.stop_utility(entropy)
                next_entropy = self.expected_next_entropy(entropy)
                continue_value = -self.continuation_cost(budget) + self.interpolate(prev_values, next_entropy)
                choose_stop = stop_value >= continue_value
                current_values.append(max(stop_value, continue_value))
                current_stop_mask.append(choose_stop)
                if choose_stop:
                    threshold = entropy
            values[budget] = current_values
            stop_mask[budget] = current_stop_mask
            thresholds[budget] = threshold if threshold is not None else -1.0
        return OracleResult(
            entropy_grid=self.entropy_grid,
            values=values,
            stop_mask=stop_mask,
            thresholds=thresholds,
        )
