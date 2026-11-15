#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# No model/data download, credential initialization, or experiment env sourcing.
test -s "$TASK_ROOT/workspace/vendor/megatron/megatron/core/models/gpt/gpt_model.py"
test -s "$TASK_ROOT/workspace/vendor/megatron/megatron/core/transformer/multi_token_prediction.py"
test -s "$TASK_ROOT/workspace/vendor/miles/miles/backends/megatron_utils/model.py"
echo '[prepare] staged complete framework packages; GPU compatibility requires bounded smoke'
