#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"
export WANDB_MODE=offline
export WANDB_PROJECT=autorlbench-reproductions
export WANDB_DIR="$PWD/results/wandb"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
mkdir -p "$WANDB_DIR"
python -m torch.distributed.run --standalone --nproc_per_node=1 run.py \
  --model model --data data --output results --case normalization --steps 24 "$@"
