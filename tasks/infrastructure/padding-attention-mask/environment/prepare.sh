#!/usr/bin/env bash
set -euo pipefail
ENV_DIR="${AUTORLBENCH_ENVIRONMENT:-$(cd "$(dirname "$0")" && pwd)}"
WORKSPACE="${AUTORLBENCH_WORKSPACE:-$(cd "$ENV_DIR/../workspace" && pwd)}"
export CUDA_VISIBLE_DEVICES=""
# The pinned harness supplies fetch_hf.py with each task, not a global helper.
python3 "$ENV_DIR/prepare_fetch.py" --manifest "$ENV_DIR/model_assets.json" --dest "$WORKSPACE/models/student"
python3 "$ENV_DIR/prepare_fetch.py" --manifest "$ENV_DIR/dapo_assets.json" --dest "$WORKSPACE/datasets/raw/dapo"
python3 "$ENV_DIR/prepare_fetch.py" --manifest "$ENV_DIR/aime24_assets.json" --dest "$WORKSPACE/datasets/raw/aime24"
python3 "$ENV_DIR/prepare_math.py" --dapo "$WORKSPACE/datasets/raw/dapo/data/dapo-math-17k.parquet" --aime "$WORKSPACE/datasets/raw/aime24/data/train-00000-of-00001.parquet" --model "$WORKSPACE/models/student" --output "$WORKSPACE/datasets/math" --seed 42 --max-prompt-tokens 1024
