#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-seed}"
INPUT_JSONL="${INPUT_JSONL:-$PROJECT_ROOT/outputs/awm_qwen35_rollouts_8_per_task/skill_extract.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/awm_qwen35_sft_data}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.1}"
SEED="${SEED:-2026}"
OVERWRITE="${OVERWRITE:-false}"
if [[ "$OVERWRITE" != "true" && ( -e "$OUTPUT_DIR/sft_episode_skill_train.parquet" || -e "$OUTPUT_DIR/sft_episode_skill_val.parquet" ) ]]; then
  echo "Output already exists: $OUTPUT_DIR (set OVERWRITE=true to replace)" >&2
  exit 1
fi
if [[ "$OVERWRITE" == "true" ]]; then
  rm -f "$OUTPUT_DIR"/sft_episode_skill_{all,train,val}.{jsonl,parquet}
fi
cd "$PROJECT_ROOT"
conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/convert_skill_extract.py" --input "$INPUT_JSONL" --output-dir "$OUTPUT_DIR" --val-ratio "$SFT_VAL_RATIO" --seed "$SEED"
