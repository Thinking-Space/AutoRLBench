#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${AUTORLBENCH_PREPARED:+${AUTORLBENCH_PREPARED}/pysite:}$PWD/vendor/verl:$PWD"
export WANDB_MODE=offline
export WANDB_DIR="${LOG_DIR:-$PWD/new_run}"
mkdir -p "$WANDB_DIR"
python3 -m reproductions.sql_credit.train --model "$PWD/assets/model" --steps 3 --seed 17 --output "$WANDB_DIR"
