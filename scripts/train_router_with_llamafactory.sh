#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_router_with_llamafactory.sh
# Optional env vars:
#   LLAMAFACTORY_CONFIG=training/llamafactory_configs/qwen25_router_lora_sft.yaml
#   CUDA_VISIBLE_DEVICES=0

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python3 scripts/export_router_sft_data.py \
  --tasks data/realistic_mixed_task_benchmark_v3.json \
  --output training/llamafactory_data/router_sft_seed.jsonl

CONFIG_PATH="${LLAMAFACTORY_CONFIG:-training/llamafactory_configs/qwen25_router_lora_sft.yaml}"

echo "[INFO] Training with config: $CONFIG_PATH"
llamafactory-cli train "$CONFIG_PATH"
