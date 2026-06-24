#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
OUT="training/llamafactory_data/router_sft_mixed.jsonl"
cat \
  training/llamafactory_data/router_sft_seed.jsonl \
  training/llamafactory_data/router_sft_tau2_airline_raw.jsonl \
  training/llamafactory_data/router_sft_bfcl_raw.jsonl \
  > "$OUT"
python3 - <<'PY'
from pathlib import Path
p=Path('training/llamafactory_data/router_sft_mixed.jsonl')
print({'output':str(p),'num_examples':sum(1 for _ in p.open())})
PY
