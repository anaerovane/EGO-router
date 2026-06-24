from __future__ import annotations

import statistics
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.synthetic_entropy_env import EnvConfig, SyntheticEntropyEnv
from src.policies.baselines import (
    BudgetAwareThresholdPolicy,
    FixedDepthPolicy,
    FixedThresholdPolicy,
    ImmediateStopPolicy,
    NeverStopEarlyPolicy,
    OracleThresholdPolicy,
)
from src.policies.ego_threshold import EGOEntropyGatePolicy
from src.solvers.dp_oracle import DynamicProgrammingOracle


def evaluate_policy(env_config: EnvConfig, policy, num_episodes: int = 2000, seed: int = 0):
    env = SyntheticEntropyEnv(env_config, seed=seed)
    rewards = []
    stopping_times = []
    final_entropies = []
    for episode_idx in range(num_episodes):
        result = env.simulate_episode(policy)
        rewards.append(result["total_reward"])
        trajectory = result["trajectory"]
        num_continue_actions = sum(1 for step in trajectory if step['action'] == 'continue')
        stopping_times.append(num_continue_actions)
        final_entropies.append(result["final_observation"]["latent_entropy"])
        env.rng.seed(seed + episode_idx + 1)
    return {
        "avg_reward": statistics.mean(rewards),
        "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "avg_stopping_time": statistics.mean(stopping_times),
        "avg_final_entropy": statistics.mean(final_entropies),
    }


def main():
    config = EnvConfig(
        max_entropy=1.0,
        budget=8,
        rho=0.25,
        gamma=0.7,
        process_noise=0.03,
        observation_noise=0.02,
        alpha=0.8,
        continuation_cost=0.06,
        scarcity_cost_scale=0.08,
        init_entropy_low=0.1,
        init_entropy_high=0.95,
    )

    oracle = DynamicProgrammingOracle(config, grid_size=301).solve()

    policies = {
        "immediate_stop": ImmediateStopPolicy(),
        "never_stop_early": NeverStopEarlyPolicy(),
        "fixed_depth_2": FixedDepthPolicy(continue_steps=2, initial_budget=config.budget),
        "fixed_depth_4": FixedDepthPolicy(continue_steps=4, initial_budget=config.budget),
        "fixed_threshold_0.25": FixedThresholdPolicy(threshold=0.25),
        "fixed_threshold_0.40": FixedThresholdPolicy(threshold=0.40),
        "budget_threshold": BudgetAwareThresholdPolicy(h0=0.08, beta=0.75),
        "ego_entropy_gate": EGOEntropyGatePolicy(h0=0.08, alpha_h=0.75),
        "oracle_threshold": OracleThresholdPolicy(thresholds=oracle.thresholds),
    }

    print("=== Synthetic Theorem-A Validation ===")
    print("Oracle thresholds by remaining budget:")
    for budget in sorted(oracle.thresholds):
        print(f"  b={budget}: h*={oracle.thresholds[budget]:.4f}")
    print()

    results = {}
    for name, policy in policies.items():
        metrics = evaluate_policy(config, policy, num_episodes=1500, seed=42)
        results[name] = metrics

    sorted_results = sorted(results.items(), key=lambda kv: kv[1]["avg_reward"], reverse=True)
    print("Policy results (sorted by avg_reward):")
    for name, metrics in sorted_results:
        print(
            f"- {name:20s} | "
            f"reward={metrics['avg_reward']:.4f} +/- {metrics['std_reward']:.4f} | "
            f"stop_time={metrics['avg_stopping_time']:.2f} | "
            f"final_H={metrics['avg_final_entropy']:.4f}"
        )


if __name__ == "__main__":
    main()
