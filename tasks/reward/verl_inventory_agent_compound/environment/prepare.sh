#!/usr/bin/env bash
# Prepare the recorded rows and pinned model. The local data-only smoke check
# can opt out of model preparation with PREPARE_TRAINING_ASSETS=0.
set -euo pipefail

: "${AUTORLBENCH_WORKSPACE:?set the staged workspace path}"
task_env="$(cd "$(dirname "$0")" && pwd)"
python3 "$task_env/prepare_data.py"

if [[ "${PREPARE_TRAINING_ASSETS:-1}" == "1" ]]; then
    python3 "$task_env/prepare_data.py" --model
fi
