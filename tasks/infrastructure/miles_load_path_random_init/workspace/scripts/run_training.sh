#!/usr/bin/env bash
# Training launch script (defective version)
# MEGATRON_MODEL_PATH points to release/ subdirectory instead of parent.
# This causes Megatron to silently skip checkpoint loading.

set -ex

# BUG: MEGATRON_MODEL_PATH points to release/ subdirectory.
# latest_checkpointed_iteration.txt lives in the PARENT directory, not here.
# Megatron will log "will not load any checkpoints and will start from random"
# and continue training with random weights.
export MEGATRON_MODEL_PATH="${MEGATRON_MODEL_PATH:-/models/Qwen3.5-9B_torch_dist/release}"

# Correct path would be:
# export MEGATRON_MODEL_PATH="${MEGATRON_MODEL_PATH:-/models/Qwen3.5-9B_torch_dist}"

python3 -m miles.train \
  --hf-checkpoint /models/Qwen3.5-9B \
  --load "${MEGATRON_MODEL_PATH}" \
  --finetune --no-load-optim --no-load-rng \
  --num-rollout 200 \
  --rollout-batch-size 16 \
  --n-samples-per-prompt 8 \
  "$@"
