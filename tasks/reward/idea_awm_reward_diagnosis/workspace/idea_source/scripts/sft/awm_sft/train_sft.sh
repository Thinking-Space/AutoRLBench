#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET_NAME="awm"
DATASET_LABEL="AWM"
MODEL_BASENAME="/local/user/Qwen/Qwen3-8B"
MODELS_ROOT="/local/user/models"
# MODEL_BASENAME="Qwen3.5-9B"
# MODELS_ROOT="/local/user"
MODEL_TAG="qwen3_8b"
DEFAULT_SFT_CONDA_ENV="seed"
DEFAULT_LR="1e-6"
DEFAULT_MAX_LENGTH="8192"
MULTIMODAL="false"
DATA_DIR_EXTRA_SUFFIX=""
OUTPUT_DIR=/local/user/models/sft/awm_episode_skill_sft_qwen3_8b_ep3_$(date +%Y%m%d_%H%M%S) 
source "$PROJECT_ROOT/scripts/sft/_common_v1/trainer.sh"
