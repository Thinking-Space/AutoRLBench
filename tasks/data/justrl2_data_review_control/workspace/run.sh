#!/usr/bin/env bash
set -euo pipefail
: "${AUTORLBENCH_PREPARED:?}" "${AUTORLBENCH_RESULTS:?}"
exec python3 "$AUTORLBENCH_PREPARED/incident_engine/run.py" \
  --scene "$AUTORLBENCH_PREPARED/incident" \
  --prepared "$AUTORLBENCH_PREPARED/assets" \
  --native "$AUTORLBENCH_PREPARED/incident/checkpoints/actor" \
  --loader "$PWD/loader.py" --output "$AUTORLBENCH_RESULTS/experiment_logs/run-home"
