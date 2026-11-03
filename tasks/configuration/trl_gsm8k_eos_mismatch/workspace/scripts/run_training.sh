#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ ! -f model/model.safetensors ]; then
    printf '%s
' 'Historical diagnosis snapshot: weights and training dataset are not distributed. See workspace README.' >&2
    exit 1
fi
export INIT_MODEL="${INIT_MODEL:-model}"
export OUT_DIR="${OUT_DIR:-outputs/checkpoint}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-outputs/rollouts}"
export QWEN3_TEMPLATE="${QWEN3_TEMPLATE:-templates/qwen3.jinja}"
exec accelerate launch --config_file "$(python3 -c 'import pathlib, trl; print(pathlib.Path(trl.__file__).parent / "accelerate_configs/multi_gpu.yaml")')" --num_processes 8 scripts/train_grpo.py
