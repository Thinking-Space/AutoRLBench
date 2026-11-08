#!/usr/bin/env bash
set -euo pipefail
W="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${RL_PYTHON:-}" ]]; then
    RUN_PYTHON="$RL_PYTHON"
elif [[ -n "${AUTORLBENCH_PREPARED:-}" && -x "$AUTORLBENCH_PREPARED/runtime/bin/python" ]]; then
    RUN_PYTHON="$AUTORLBENCH_PREPARED/runtime/bin/python"
else
    RUN_PYTHON=python3
fi
export PYTHONPATH="$W/scripts:$W/verl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec "$RUN_PYTHON" -u "$W/scripts/runner.py" --workspace . --run "${RUN_NAME:-run_003}" \
    --mode train --steps "${TRAIN_STEPS:-40}" --gpus 1 "$@"
