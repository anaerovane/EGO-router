# Synthetic Sweep Findings

## Date
2026-06-24

## What was added

Implemented a configurable synthetic sweep runner:

- `scripts/run_synthetic_sweeps.py`

The runner evaluates entropy-gated stopping across a grid of synthetic regimes and writes:

- `outputs/synthetic_sweep_results.json`
- `outputs/synthetic_sweep_summary.csv`
- `outputs/synthetic_sweep_policy_metrics.csv`

A small helper script was also added for quick console summaries:

- `scripts/summarize_synthetic_sweep.py`

---

## Sweep configuration used for the first full run

The first full run used:

- episodes per setting: `200`
- budget: `8`
- continuation costs: `0.03, 0.06, 0.09, 0.12`
- scarcity-cost scales: `0.00, 0.04, 0.08, 0.16`
- observation noises: `0.00, 0.02, 0.06`
- initial entropy regimes: `wide`, `high`
- fixed-threshold grid: `0.05` to `0.95` with step `0.05`
- tuned EGO grid:
  - `h0`: `0.00, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48`
  - `alpha_h`: `0.25, 0.50, 0.75, 1.00, 1.25, 1.50`
- DP oracle grid size: `301`

Command:

```bash
conda run -n base python scripts/run_synthetic_sweeps.py --episodes 200
```

---

## Aggregate results

Across `96` synthetic settings:

| Metric | Mean | Min | Max | Positive settings |
|---|---:|---:|---:|---:|
| Default EGO - best fixed threshold | `-0.0325` | `-0.1296` | `0.0017` | `9 / 96` |
| Tuned EGO - best fixed threshold | `0.0004` | `-0.0014` | `0.0027` | `85 / 96` |
| Oracle - tuned EGO | `0.0002` | `-0.0006` | `0.0037` | `52 / 96` |
| Tuned EGO - best fixed depth | `0.0156` | `0.0002` | `0.0340` | `96 / 96` |

---

## Main interpretation

### 1. The fixed default EGO gate is not robust enough

The default policy `ego_entropy_gate(h0=0.08, alpha_h=0.75)` is too conservative in high-cost regimes. It keeps continuing when continuation is expensive, causing a large gap against the best fixed threshold.

This is useful scientifically because it shows that the sweep is not merely confirming the original hand-picked setting. It surfaces a real tuning problem in the current parameterization.

### 2. Tuned EGO essentially matches the DP oracle in this synthetic environment

After expanding the EGO gate grid, tuned EGO is very close to the DP oracle:

- mean oracle gap: about `0.0002`
- max oracle gap: about `0.0037`

This suggests that the budget-dependent threshold form is expressive enough for this scalar entropy environment.

### 3. Tuned EGO is consistently better than fixed-depth baselines

Tuned EGO beats the best fixed-depth baseline in all `96 / 96` settings, with mean reward gain around `0.0156`.

This is currently the cleanest positive experimental claim from the synthetic sweep.

### 4. Tuned EGO only marginally beats best fixed-threshold baselines

Tuned EGO beats the best fixed threshold in `85 / 96` settings, but the average gain is small: about `0.0004`.

This means the current scalar environment is still too simple to strongly separate budget-aware thresholding from a well-tuned global fixed threshold. The result is not negative, but it is not yet a strong paper claim by itself.

---

## Best and worst regimes

### Strongest tuned-EGO gains over fixed threshold

The largest observed gains are small, around `0.0017` to `0.0027`, mostly in low-cost settings with higher observation noise and high initial entropy.

Examples:

- `cost=0.030`, `scarcity=0.040`, `obs_noise=0.060`, `init=high`: `+0.0027`
- `cost=0.030`, `scarcity=0.000`, `obs_noise=0.060`, `init=high`: `+0.0025`
- `cost=0.030`, `scarcity=0.160`, `obs_noise=0.020`, `init=wide`: `+0.0019`

### Weakest tuned-EGO regimes

The worst gap is around `-0.0014`:

- `cost=0.030`, `scarcity=0.000`, `obs_noise=0.020`, `init=wide`: `-0.0014`

This is small enough that it may be sampling noise, but it should be checked with more episodes before claiming dominance.

---

## Scientific takeaway

The sweep supports three claims:

1. Budget-aware threshold policies can closely approximate the DP oracle in the theorem-aligned scalar entropy environment.
2. Tuned EGO reliably outperforms fixed-depth continuation heuristics.
3. The current scalar environment does not yet create a large gap against the best tuned fixed-threshold baseline.

The third point is important: for the paper, the next synthetic environment should make budget heterogeneity or episode-level cost heterogeneity stronger, so that a single global fixed threshold cannot mimic the budget-aware policy so easily.

---

## Recommended next experiment

Add a second synthetic setting where the same policy faces mixed regimes within one evaluation run:

- variable per-episode continuation costs;
- variable per-episode budget sizes;
- variable scarcity-cost profiles;
- or task families with different entropy reduction curves.

The key goal is to prevent a single fixed threshold from being globally optimal across all episodes. That is the regime where EGO's budget-dependent threshold should have a clearer advantage.
