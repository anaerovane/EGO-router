# Router Fine-tuning Starter Pack

## Files
- `scripts/export_router_sft_data.py`: export seed SFT data from the local benchmark.
- `training/llamafactory_data/dataset_info.json`: LLaMA-Factory dataset registry.
- `training/llamafactory_configs/qwen25_router_lora_sft.yaml`: LoRA SFT config for Qwen2.5-1.5B-Instruct.
- `scripts/train_router_with_llamafactory.sh`: one-command export + train entrypoint.

## Quick start
```bash
python3 scripts/export_router_sft_data.py \
  --tasks data/realistic_mixed_task_benchmark_v3.json \
  --output training/llamafactory_data/router_sft_seed.jsonl

llamafactory-cli train training/llamafactory_configs/qwen25_router_lora_sft.yaml
```

Or:
```bash
bash scripts/train_router_with_llamafactory.sh
```

## Important note
This seed dataset is intentionally synthetic and should be treated as a warm-start dataset.
For real gains, replace or augment it with trajectory-derived labels exported from real agent states.

## Converting benchmark exports into router SFT
You can convert trajectory exports from external benchmarks into the same Alpaca-format SFT data.

### Supported adapter templates
- `training/benchmark_adapter_templates/bfcl_router_adapter.json`
- `training/benchmark_adapter_templates/tau2_router_adapter.json`
- `training/benchmark_adapter_templates/openhands_router_adapter.json`
- `training/benchmark_adapter_templates/gaia_router_adapter.json`

### Expected input
The converter accepts `.jsonl` or `.json` exports. It is tolerant to schema variation via field aliases.
Recommended fields per record:
- `query` / `instruction`
- `best_action` or `action_scores`
- `should_stop`
- `available_actions`
- optional `state.entropy`, `state.margin`, `state.disagreement`, `state.verifier_confidence`, `state.steps_remaining`, `state.current_best_score`
- optional `best_candidate`, `second_candidate`, `recent_candidates`

### Example
```bash
python3 scripts/convert_benchmark_logs_to_router_sft.py \
  --input /path/to/tau2_export.jsonl \
  --adapter training/benchmark_adapter_templates/tau2_router_adapter.json \
  --output training/llamafactory_data/router_sft_tau2.jsonl
```

To mix with the seed data, concatenate jsonl files:
```bash
cat training/llamafactory_data/router_sft_seed.jsonl \
    training/llamafactory_data/router_sft_tau2.jsonl \
  > training/llamafactory_data/router_sft_mixed.jsonl
```

Then add a new dataset entry to `training/llamafactory_data/dataset_info.json` or replace `router_sft_seed.jsonl`.
