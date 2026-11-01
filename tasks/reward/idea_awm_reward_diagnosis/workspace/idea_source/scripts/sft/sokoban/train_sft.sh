#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET_NAME="sokoban"
DATASET_LABEL="Sokoban"
MODEL_BASENAME="Qwen2.5-VL-3B-Instruct"
MODEL_TAG="qwen25_vl_3b"
DEFAULT_SFT_CONDA_ENV=""
DEFAULT_LR="2e-6"
DEFAULT_MAX_LENGTH="4096"
MULTIMODAL="true"

# shellcheck source=../_common/trainer.sh
source "$PROJECT_ROOT/scripts/sft/_common/trainer.sh"
