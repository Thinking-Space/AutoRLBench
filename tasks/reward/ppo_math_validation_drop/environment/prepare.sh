#!/usr/bin/env bash
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_ENVIRONMENT:?}"
: "${AUTORLBENCH_PREPARED:?}"

# Validate the packaged evidence before setting up the training inputs.
test -s "$AUTORLBENCH_WORKSPACE/evidence/step500_responses.jsonl"
test -s "$AUTORLBENCH_WORKSPACE/evidence/trainer_metrics.log"
python3 "$AUTORLBENCH_ENVIRONMENT/prepare_runtime.py"
printf 'PYTHONPATH=%s/pysite\n' "$AUTORLBENCH_PREPARED" > "$AUTORLBENCH_PREPARED/env"
export PYTHONPATH="$AUTORLBENCH_PREPARED/pysite${PYTHONPATH:+:$PYTHONPATH}"
python3 "$AUTORLBENCH_ENVIRONMENT/prepare_data.py"
