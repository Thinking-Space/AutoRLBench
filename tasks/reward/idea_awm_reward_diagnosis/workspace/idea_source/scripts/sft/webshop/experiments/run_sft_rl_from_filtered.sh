#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
# shellcheck source=../../_common/teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft/_common/teacher_naming.sh"

MODELS_ROOT="${MODELS_ROOT:?Please set MODELS_ROOT in $ENV_FILE.}"

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/outputs/webshop_episode_skill_pipeline_filtered_qwen25_3b_${SFT_SELF_DIR_SUFFIX}}"
SFT_BASE_MODEL_PATH="${SFT_BASE_MODEL_PATH:-$MODELS_ROOT/Qwen2.5-3B-Instruct}"
SFT_EXPORT_MODEL_DIR="${SFT_EXPORT_MODEL_DIR:-$MODELS_ROOT/Qwen2.5-3B-Instruct-webshop-episode-skill-sft-${SFT_SELF_SUFFIX}}"
SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-3}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-12288}"
SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-8}"
SFT_MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-1}"

RL_SCRIPT="${RL_SCRIPT:-$PROJECT_ROOT/examples/idea_trainer/run_webshop_sft_glm_self.sh}"
RL_EXPERIMENT_NAME="${RL_EXPERIMENT_NAME:-seed-grpo_qwen2.5_3b_webshop_episode_no_skill_loss_sft_filtered_policy-vllm}"
RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-$MODELS_ROOT/ckpt/$RL_EXPERIMENT_NAME}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Running WebShop filtered SFT then RL"
echo "  data dir:        $DATA_DIR"
echo "  sft base model:  $SFT_BASE_MODEL_PATH"
echo "  sft export dir:  $SFT_EXPORT_MODEL_DIR"
echo "  sft max length:  $SFT_MAX_LENGTH"
echo "  rl script:       $RL_SCRIPT"
echo "  rl experiment:   $RL_EXPERIMENT_NAME"

DATA_DIR="$DATA_DIR" \
MODEL_PATH="$SFT_BASE_MODEL_PATH" \
TOTAL_EPOCHS="$SFT_TOTAL_EPOCHS" \
MAX_LENGTH="$SFT_MAX_LENGTH" \
TRAIN_BATCH_SIZE="$SFT_TRAIN_BATCH_SIZE" \
MICRO_BATCH_SIZE_PER_GPU="$SFT_MICRO_BATCH_SIZE_PER_GPU" \
EXPORT_MODEL_DIR="$SFT_EXPORT_MODEL_DIR" \
bash "$PROJECT_ROOT/scripts/sft/webshop/train_sft.sh"

HF_MODEL_PATH="$SFT_EXPORT_MODEL_DIR" \
MODEL_PATH="$SFT_EXPORT_MODEL_DIR" \
EXPERIMENT_NAME="$RL_EXPERIMENT_NAME" \
DEFAULT_LOCAL_DIR="$RL_OUTPUT_DIR" \
bash "$RL_SCRIPT"
