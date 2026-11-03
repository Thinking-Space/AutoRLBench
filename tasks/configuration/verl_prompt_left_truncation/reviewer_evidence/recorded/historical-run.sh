#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p outputs/wandb
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export WANDB_DIR="$PWD/outputs/wandb"
export VERL_FILE_LOGGER_PATH="$PWD/outputs/metrics.jsonl"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${SEED:-42}"
export VLLM_USE_V1=1
python -m case.launch "$@" 2>&1 | tee outputs/train.log
