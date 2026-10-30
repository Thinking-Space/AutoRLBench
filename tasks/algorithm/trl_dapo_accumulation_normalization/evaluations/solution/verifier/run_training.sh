#!/usr/bin/env bash
set -euo pipefail
cd "$AUTORLBENCH_WORKSPACE"
export PYTHONPATH="$PWD/src"
export WANDB_MODE=offline
export WANDB_PROJECT=autorlbench-reproductions
export WANDB_DIR="$AUTORLBENCH_RESULTS/experiment_logs/wandb"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
mkdir -p "$WANDB_DIR"
python3 -m torch.distributed.run --standalone --nproc_per_node=1 "$AUTORLBENCH_VERIFIER/resources/train.py" \
  --model model --data data --output "$AUTORLBENCH_RESULTS/experiment_logs" --case normalization --steps 24 \
  2>&1 | tee "$AUTORLBENCH_RESULTS/experiment_logs/train.log"
