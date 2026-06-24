from __future__ import annotations

import csv
from pathlib import Path


SUMMARY_PATH = Path("outputs/synthetic_sweep_summary.csv")


def main() -> None:
    rows = list(csv.DictReader(SUMMARY_PATH.open(encoding="utf-8")))
    print("settings", len(rows))
    keys = [
        "ego_minus_best_fixed_threshold",
        "tuned_ego_minus_best_fixed_threshold",
        "oracle_minus_tuned_ego",
        "tuned_ego_minus_best_fixed_depth",
    ]
    for key in keys:
        values = [float(row[key]) for row in rows]
        print(
            key,
            "mean",
            sum(values) / len(values),
            "min",
            min(values),
            "max",
            max(values),
            "positive",
            sum(value > 0 for value in values),
        )

    print("worst tuned vs fixed:")
    for row in sorted(rows, key=lambda item: float(item["tuned_ego_minus_best_fixed_threshold"]))[:5]:
        print(
            row["setting_id"],
            row["tuned_ego"],
            row["best_fixed_threshold"],
            row["tuned_ego_minus_best_fixed_threshold"],
            row["oracle_minus_tuned_ego"],
        )

    print("best tuned vs fixed:")
    for row in sorted(rows, key=lambda item: float(item["tuned_ego_minus_best_fixed_threshold"]), reverse=True)[:5]:
        print(
            row["setting_id"],
            row["tuned_ego"],
            row["best_fixed_threshold"],
            row["tuned_ego_minus_best_fixed_threshold"],
            row["oracle_minus_tuned_ego"],
        )


if __name__ == "__main__":
    main()
