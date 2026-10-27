#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

: "${MODELS_ROOT:?Please set MODELS_ROOT in $ENV_FILE or the environment.}"

# shellcheck source=../_common/teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft/_common/teacher_naming.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PIPELINE_BASE_NAME="${PIPELINE_BASE_NAME:-sokoban_episode_skill_pipeline_qwen25_vl_3b_${SFT_SELF_DIR_SUFFIX}}"
PIPELINE_ROOT="${PIPELINE_ROOT:-$PROJECT_ROOT/outputs/$PIPELINE_BASE_NAME}"
DATA_DIR="${DATA_DIR:-$PIPELINE_ROOT}"
LOG_DIR="${LOG_DIR:-$PIPELINE_ROOT/logs}"
SFT_EXPORT_MODEL_DIR="${SFT_EXPORT_MODEL_DIR:-$MODELS_ROOT/Qwen2.5-VL-3B-Instruct-sokoban-episode-skill-sft-${SFT_SELF_SUFFIX}}"
mkdir -p "$LOG_DIR"

echo "Stage 1/3: preparing Sokoban SFT data"
PIPELINE_ROOT="$PIPELINE_ROOT" DATA_DIR="$DATA_DIR" \
    bash "$SCRIPT_DIR/prepare_data.sh"

if [[ "${RUN_SFT:-true}" == "true" ]]; then
    echo "Stage 2/3: training Sokoban SFT model"
    DATA_DIR="$DATA_DIR" \
    EXPORT_MODEL_DIR="$SFT_EXPORT_MODEL_DIR" \
    TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-3}" \
        bash "$SCRIPT_DIR/train_sft.sh" \
        2>&1 | tee "$LOG_DIR/sft.log"
fi

if [[ "${RUN_RL:-true}" == "true" ]]; then
    if [[ ! -f "$SFT_EXPORT_MODEL_DIR/config.json" ]]; then
        echo "SFT model not found: $SFT_EXPORT_MODEL_DIR" >&2
        exit 1
    fi

    echo "Stage 3/3: running Sokoban SEED RL"
    SEED_RL_EXPERIMENT_NAME="${SEED_RL_EXPERIMENT_NAME:-seed_qwen2.5_vl_3b_sokoban_sft_${SFT_SELF_DIR_SUFFIX}_$RUN_ID}"
    SEED_RL_DEFAULT_LOCAL_DIR="${SEED_RL_DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$SEED_RL_EXPERIMENT_NAME}"
    HF_MODEL_PATH="$SFT_EXPORT_MODEL_DIR" \
    EXPERIMENT_NAME="$SEED_RL_EXPERIMENT_NAME" \
    DEFAULT_LOCAL_DIR="$SEED_RL_DEFAULT_LOCAL_DIR" \
    SOKOBAN_SAVE_IMAGES=True \
        bash "$PROJECT_ROOT/examples/idea_trainer/run_sokoban_sft_gemini_self.sh" \
        2>&1 | tee "$LOG_DIR/seed_rl.log"
fi

echo "Sokoban pipeline finished."
