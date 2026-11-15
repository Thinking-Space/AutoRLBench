#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_WORKSPACE="${AUTORLBENCH_WORKSPACE:-$TASK_DIR/workspace}"
PREPARED_ROOT="${AUTORLBENCH_PREPARED:?prepare requires the harness prepared directory}"
# These are historical Miles dependencies absent from the public base image.
# Exact wheel SHA256 and byte sizes are declared in dependencies.json.
python3 -B "$ENVIRONMENT_DIR/prepare_dependencies.py" "$PREPARED_ROOT/dependencies"
printf 'PYTHONPATH=%s\n' "$PREPARED_ROOT/dependencies/python" >> "$PREPARED_ROOT/env"
# No model/data downloads. Public Qwen2 architecture, random seed 20261115.
# Native source pins and image/platform alias are in runtime_sources.json.
# Compiled CUDA/TE/FlashAttention/SGLang dependencies must come from that image;
# bundled MCore 0.18/Bridge overrides its older Python source.
PYTHONDONTWRITEBYTECODE=1 python3 -B "$TASK_WORKSPACE/scripts/prepare_public_assets.py" \
  --dest "$TASK_WORKSPACE/assets/public-qwen2"
