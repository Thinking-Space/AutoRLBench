#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${MODEL_PATH:?Please set MODEL_PATH to Qwen3-4B-Base}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RAY_RESTART="${RAY_RESTART:-0}"
export RUN_MODE=ppo_critic
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export NNODES=1

export LR=1e-6
export PROJECT_NAME=gurumath-smoke
export CKPTS_DIR="$ROOT/autorlbench_case/output_ckpts"
export LOGGER=console
export SMOKE_SCORING=1

export TRAIN_PROMPT_BSZ=16
export GEN_PROMPT_BSZ=16
export TRAIN_PROMPT_MINI_BSZ=16
export N_RESP_PER_PROMPT=8

export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=1024
export CRITIC_MICRO_BSZ_PER_GPU=1
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.25

export TRAIN_FILE="$ROOT/autorlbench_case/assets/train_1024.parquet"
export VAL_AIME_FILE="$ROOT/autorlbench_case/assets/aime32.parquet"
export VAL_MATH500_FILE="$ROOT/autorlbench_case/assets/math500_32.parquet"
export VAL_FILES="['$VAL_AIME_FILE','$VAL_MATH500_FILE']"

bash "$ROOT/recipe/vimpo/run_vimpo_critic.sh" \
  trainer.total_training_steps=5 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  trainer.val_before_train=True
