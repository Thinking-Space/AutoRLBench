#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_MODE=offline
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$PWD/vendor${PYTHONPATH:+:$PYTHONPATH}"
exec python shared_runtime.py --config config.shared.json --log-dir "${LOG_DIR:-experiment_logs}" "$@"
