#!/usr/bin/env bash
# Training launch script (defective version)
# ROLLOUT_NUM_GPUS_PER_ENGINE=2 causes sglang TP=2 for Qwen3.5.
# sglang 0.5.9 produces garbled HTTP 200 output for Qwen3.5 at TP>1 (issue 21039).

set -ex

# BUG: ROLLOUT_NUM_GPUS_PER_ENGINE=2 causes sglang TP=2 for Qwen3.5.
# sglang 0.5.9 returns HTTP 200 with garbled content for Qwen3.5 at TP>1.
# Fix: set ROLLOUT_NUM_GPUS_PER_ENGINE=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}

python3 -m miles.train \
  --hf-checkpoint /models/Qwen3.5-9B \
  --load /models/Qwen3.5-9B_torch_dist \
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE} \
  --num-rollout 200 \
  --rollout-batch-size 16 \
  "$@"
