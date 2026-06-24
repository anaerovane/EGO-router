from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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


INIT_REGIMES = {
    "wide": (0.10, 0.95),
    "medium_high": (0.35, 0.95),
    "high": (0.60, 0.98),
}


FIXED_DEPTHS = (1, 2, 4, 6)


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def frange(start: float, stop: float, step: float) -> List[float]:
    values = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 10))
        current += step
    return values


def build_policies(
    config: EnvConfig,
    oracle_thresholds: Dict[int, float],
    fixed_thresholds: Sequence[float],
    ego_h0_values: Sequence[float],
    ego_alpha_values: Sequence[float],
):
    policies = {
        "immediate_stop": ImmediateStopPolicy(),
        "never_stop_early": NeverStopEarlyPolicy(),
        "budget_threshold": BudgetAwareThresholdPolicy(h0=0.08, beta=0.75),
        "ego_entropy_gate": EGOEntropyGatePolicy(h0=0.08, alpha_h=0.75),
        "oracle_threshold": OracleThresholdPolicy(thresholds=oracle_thresholds),
    }
    for h0 in ego_h0_values:
        for alpha_h in ego_alpha_values:
            name = f"ego_grid_h0={h0:.2f}_alpha={alpha_h:.2f}"
            if name == "ego_grid_h0=0.08_alpha=0.75":
                continue
            policies[name] = EGOEntropyGatePolicy(h0=h0, alpha_h=alpha_h)
    for depth in FIXED_DEPTHS:
        if depth <= config.budget:
            policies[f"fixed_depth_{depth}"] = FixedDepthPolicy(
                continue_steps=depth,
                initial_budget=config.budget,
            )
    for threshold in fixed_thresholds:
        policies[f"fixed_threshold_{threshold:.2f}"] = FixedThresholdPolicy(
            threshold=threshold,
        )
    return policies


def evaluate_policy(env_config: EnvConfig, policy, num_episodes: int, seed: int):
    rewards = []
    stopping_times = []
    final_entropies = []
    stop_budget_counts = {budget: 0 for budget in range(env_config.budget + 1)}

    for episode_idx in range(num_episodes):
        env = SyntheticEntropyEnv(env_config, seed=seed + episode_idx)
        result = env.simulate_episode(policy)
        trajectory = result["trajectory"]
        continue_actions = sum(1 for step in trajectory if step["action"] == "continue")
        rewards.append(result["total_reward"])
        stopping_times.append(continue_actions)
        final_entropies.append(result["final_observation"]["latent_entropy"])

        stop_budget = int(result["final_observation"]["remaining_budget"])
        if trajectory and trajectory[-1]["action"] == "stop":
            stop_budget = int(trajectory[-1]["obs"]["remaining_budget"])
        stop_budget_counts[stop_budget] += 1

    stop_budget_freq = {
        str(budget): count / num_episodes for budget, count in stop_budget_counts.items()
    }
    return {
        "avg_reward": statistics.mean(rewards),
        "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "avg_stopping_time": statistics.mean(stopping_times),
        "avg_final_entropy": statistics.mean(final_entropies),
        "stop_budget_freq": stop_budget_freq,
    }


def config_id(config: EnvConfig, init_regime: str) -> str:
    return (
        f"cost={config.continuation_cost:.3f}|"
        f"scarcity={config.scarcity_cost_scale:.3f}|"
        f"obs_noise={config.observation_noise:.3f}|"
        f"init={init_regime}"
    )


def summarize_setting(policy_results: Dict[str, Dict[str, object]]) -> Dict[str, float | str]:
    fixed_threshold_items = [
        (name, metrics)
        for name, metrics in policy_results.items()
        if name.startswith("fixed_threshold_")
    ]
    fixed_depth_items = [
        (name, metrics)
        for name, metrics in policy_results.items()
        if name.startswith("fixed_depth_")
    ]
    ego_grid_items = [
        (name, metrics)
        for name, metrics in policy_results.items()
        if name == "ego_entropy_gate" or name.startswith("ego_grid_")
    ]
    best_fixed_threshold = max(
        fixed_threshold_items,
        key=lambda item: item[1]["avg_reward"],
    )
    best_fixed_depth = max(
        fixed_depth_items,
        key=lambda item: item[1]["avg_reward"],
    )
    best_ego = max(
        ego_grid_items,
        key=lambda item: item[1]["avg_reward"],
    )
    ego_reward = policy_results["ego_entropy_gate"]["avg_reward"]
    tuned_ego_reward = best_ego[1]["avg_reward"]
    oracle_reward = policy_results["oracle_threshold"]["avg_reward"]
    return {
        "ego_reward": ego_reward,
        "tuned_ego": best_ego[0],
        "tuned_ego_reward": tuned_ego_reward,
        "oracle_reward": oracle_reward,
        "best_fixed_threshold": best_fixed_threshold[0],
        "best_fixed_threshold_reward": best_fixed_threshold[1]["avg_reward"],
        "best_fixed_depth": best_fixed_depth[0],
        "best_fixed_depth_reward": best_fixed_depth[1]["avg_reward"],
        "ego_minus_best_fixed_threshold": ego_reward
        - best_fixed_threshold[1]["avg_reward"],
        "tuned_ego_minus_best_fixed_threshold": tuned_ego_reward
        - best_fixed_threshold[1]["avg_reward"],
        "ego_minus_best_fixed_depth": ego_reward - best_fixed_depth[1]["avg_reward"],
        "tuned_ego_minus_best_fixed_depth": tuned_ego_reward
        - best_fixed_depth[1]["avg_reward"],
        "oracle_minus_ego": oracle_reward - ego_reward,
        "oracle_minus_tuned_ego": oracle_reward - tuned_ego_reward,
    }


def iter_configs(args) -> Iterable[tuple[str, EnvConfig]]:
    base = EnvConfig(
        max_entropy=1.0,
        budget=args.budget,
        rho=args.rho,
        gamma=args.gamma,
        process_noise=args.process_noise,
        alpha=args.alpha,
    )
    for continuation_cost in parse_float_list(args.continuation_costs):
        for scarcity_cost_scale in parse_float_list(args.scarcity_cost_scales):
            for observation_noise in parse_float_list(args.observation_noises):
                for init_regime in args.init_regimes:
                    low, high = INIT_REGIMES[init_regime]
                    yield init_regime, replace(
                        base,
                        continuation_cost=continuation_cost,
                        scarcity_cost_scale=scarcity_cost_scale,
                        observation_noise=observation_noise,
                        init_entropy_low=low,
                        init_entropy_high=high,
                    )


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(args) -> Dict[str, object]:
    fixed_thresholds = frange(
        args.threshold_start,
        args.threshold_stop,
        args.threshold_step,
    )
    ego_h0_values = parse_float_list(args.ego_h0_values)
    ego_alpha_values = parse_float_list(args.ego_alpha_values)
    settings = []
    policy_rows = []
    summary_rows = []

    for setting_idx, (init_regime, config) in enumerate(iter_configs(args), start=1):
        setting_name = config_id(config, init_regime)
        print(f"[{setting_idx}] {setting_name}")
        oracle = DynamicProgrammingOracle(config, grid_size=args.oracle_grid_size).solve()
        policies = build_policies(
            config,
            oracle.thresholds,
            fixed_thresholds,
            ego_h0_values,
            ego_alpha_values,
        )

        policy_results = {}
        for policy_name, policy in policies.items():
            metrics = evaluate_policy(
                config,
                policy,
                num_episodes=args.episodes,
                seed=args.seed,
            )
            policy_results[policy_name] = metrics
            policy_rows.append(
                {
                    "setting_id": setting_name,
                    "policy": policy_name,
                    "continuation_cost": config.continuation_cost,
                    "scarcity_cost_scale": config.scarcity_cost_scale,
                    "observation_noise": config.observation_noise,
                    "init_regime": init_regime,
                    "init_entropy_low": config.init_entropy_low,
                    "init_entropy_high": config.init_entropy_high,
                    "avg_reward": metrics["avg_reward"],
                    "std_reward": metrics["std_reward"],
                    "avg_stopping_time": metrics["avg_stopping_time"],
                    "avg_final_entropy": metrics["avg_final_entropy"],
                }
            )

        summary = summarize_setting(policy_results)
        summary_row = {
            "setting_id": setting_name,
            "continuation_cost": config.continuation_cost,
            "scarcity_cost_scale": config.scarcity_cost_scale,
            "observation_noise": config.observation_noise,
            "init_regime": init_regime,
            **summary,
        }
        summary_rows.append(summary_row)
        settings.append(
            {
                "setting_id": setting_name,
                "config": asdict(config),
                "init_regime": init_regime,
                "oracle_thresholds": oracle.thresholds,
                "summary": summary,
                "policies": policy_results,
            }
        )
        print(
            "    "
            f"ego_default={summary['ego_reward']:.4f}, "
            f"tuned_ego={summary['tuned_ego_reward']:.4f} "
            f"({summary['tuned_ego']}), "
            f"best_fixed_threshold={summary['best_fixed_threshold_reward']:.4f} "
            f"({summary['best_fixed_threshold']}), "
            f"tuned_gap={summary['tuned_ego_minus_best_fixed_threshold']:+.4f}, "
            f"oracle_gap={summary['oracle_minus_tuned_ego']:+.4f}"
        )

    return {
        "metadata": {
            "episodes": args.episodes,
            "seed": args.seed,
            "fixed_thresholds": fixed_thresholds,
            "fixed_depths": list(FIXED_DEPTHS),
            "ego_h0_values": ego_h0_values,
            "ego_alpha_values": ego_alpha_values,
            "oracle_grid_size": args.oracle_grid_size,
        },
        "settings": settings,
        "policy_rows": policy_rows,
        "summary_rows": summary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run synthetic sweeps for EGO entropy-gated stopping.",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--process-noise", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--continuation-costs", default="0.03,0.06,0.09,0.12")
    parser.add_argument("--scarcity-cost-scales", default="0.00,0.04,0.08,0.16")
    parser.add_argument("--observation-noises", default="0.00,0.02,0.06")
    parser.add_argument(
        "--init-regimes",
        nargs="+",
        choices=sorted(INIT_REGIMES),
        default=["wide", "high"],
    )
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--ego-h0-values", default="0.00,0.08,0.16,0.24,0.32,0.40,0.48")
    parser.add_argument("--ego-alpha-values", default="0.25,0.50,0.75,1.00,1.25,1.50")
    parser.add_argument("--oracle-grid-size", type=int, default=301)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_sweep_results.json",
    )
    parser.add_argument(
        "--summary-csv-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_sweep_summary.csv",
    )
    parser.add_argument(
        "--policy-csv-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_sweep_policy_metrics.csv",
    )
    args = parser.parse_args()

    results = run_sweep(args)
    write_json(args.json_out, {"metadata": results["metadata"], "settings": results["settings"]})
    write_csv(
        args.summary_csv_out,
        results["summary_rows"],
        [
            "setting_id",
            "continuation_cost",
            "scarcity_cost_scale",
            "observation_noise",
            "init_regime",
            "ego_reward",
            "tuned_ego",
            "tuned_ego_reward",
            "oracle_reward",
            "best_fixed_threshold",
            "best_fixed_threshold_reward",
            "best_fixed_depth",
            "best_fixed_depth_reward",
            "ego_minus_best_fixed_threshold",
            "tuned_ego_minus_best_fixed_threshold",
            "ego_minus_best_fixed_depth",
            "tuned_ego_minus_best_fixed_depth",
            "oracle_minus_ego",
            "oracle_minus_tuned_ego",
        ],
    )
    write_csv(
        args.policy_csv_out,
        results["policy_rows"],
        [
            "setting_id",
            "policy",
            "continuation_cost",
            "scarcity_cost_scale",
            "observation_noise",
            "init_regime",
            "init_entropy_low",
            "init_entropy_high",
            "avg_reward",
            "std_reward",
            "avg_stopping_time",
            "avg_final_entropy",
        ],
    )
    print(f"Wrote JSON results to {args.json_out}")
    print(f"Wrote summary CSV to {args.summary_csv_out}")
    print(f"Wrote policy CSV to {args.policy_csv_out}")


if __name__ == "__main__":
    main()
