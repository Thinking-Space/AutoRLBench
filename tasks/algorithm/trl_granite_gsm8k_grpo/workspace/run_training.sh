#!/usr/bin/env bash
set -euo pipefail
cd "$AUTORLBENCH_WORKSPACE"
logs="$AUTORLBENCH_RESULTS/experiment_logs"
checkpoints="$(mktemp -d)"
trap 'rm -rf "$checkpoints"' EXIT
mkdir -p "$logs"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python3 -u train.py --config config.json --model model --data data \
  --output "$logs" --checkpoints "$checkpoints" 2>&1 | tee "$logs/train.log"
