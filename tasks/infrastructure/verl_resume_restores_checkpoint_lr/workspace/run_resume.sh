#!/bin/bash
# Two-stage GRPO schedule on Qwen/Qwen3-4B.
#   stage 1: train at LR1 for STAGE1_STEPS steps and save a checkpoint
#   stage 2: resume from that checkpoint with the learning rate raised to LR2 and
#            continue to STEPS total steps
# Both stages use run_grpo.sh; env vars it reads (MODEL, NGPU, RESP, ...) pass through.
set -euo pipefail
WS="$(cd "$(dirname "$0")" && pwd)"
OUT=${OUT:-$WS/outputs}
STAGE1_STEPS=${STAGE1_STEPS:-2}
STEPS=${STEPS:-5}
LR1=${LR1:-1e-6}
LR2=${LR2:-2e-6}

VERL_FILE_LOGGER_PATH=$OUT/stage1/metrics.jsonl EXP=stage1 LR=$LR1 STEPS=$STAGE1_STEPS LOG_DIR=$OUT/stage1 \
  bash "$WS/run_grpo.sh" trainer.save_freq=$STAGE1_STEPS

VERL_FILE_LOGGER_PATH=$OUT/stage2/metrics.jsonl EXP=stage2 LR=$LR2 STEPS=$STEPS LOG_DIR=$OUT/stage2 \
  bash "$WS/run_grpo.sh" trainer.resume_mode=resume_path \
  trainer.resume_from_path="$OUT/stage1/checkpoints/global_step_$STAGE1_STEPS"
