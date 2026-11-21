#!/usr/bin/env bash
set -euo pipefail
: "${AUTORLBENCH_PREPARED:?}"
exec python3 -I -S "$AUTORLBENCH_PREPARED/incident_engine/check_reads.py" \
  --scene "$AUTORLBENCH_PREPARED/incident" \
  --snapshot "$AUTORLBENCH_PREPARED/snapshot.json" \
  --loader "$PWD/loader.py"
