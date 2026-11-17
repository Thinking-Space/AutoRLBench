#!/usr/bin/env bash
# miles_thinkmix_column_missing — prepare stage.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
echo "[prepare] miles_thinkmix_column_missing"

# Install pyarrow for parquet schema inspection
pip install --quiet pyarrow==17.0.0 --target "${AUTORLBENCH_PREPARED}/pysite" 2>/dev/null || true

for f in scripts/run_training.sh data/train.parquet outputs/train_metrics.csv; do
  [ -f "${AUTORLBENCH_WORKSPACE}/${f}" ] && [ -s "${AUTORLBENCH_WORKSPACE}/${f}" ] && echo "[prepare]   OK: ${f}" || { echo "[prepare] ERROR: ${f}" >&2; exit 1; }
done
echo "[prepare] Done."
