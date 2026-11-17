#!/usr/bin/env bash
# miles_load_path_random_init — prepare stage.
#
# Runs in the miles-0.1.1 shared image. This task is a configuration defect
# (wrong MEGATRON_MODEL_PATH); no model weights or datasets need downloading.
# The workspace contains the defective launch script and frozen log evidence.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"

echo "[prepare] miles_load_path_random_init"
echo "[prepare] Verifying workspace files..."

for f in \
  scripts/run_training.sh \
  outputs/driver.log \
  outputs/reward_metrics.csv
do
  path="${AUTORLBENCH_WORKSPACE}/${f}"
  if [ ! -f "${path}" ]; then
    echo "[prepare] ERROR: missing expected file: ${f}" >&2
    exit 1
  fi
  if [ ! -s "${path}" ]; then
    echo "[prepare] ERROR: empty file: ${f}" >&2
    exit 1
  fi
  echo "[prepare]   OK: ${f}"
done

echo "[prepare] All workspace files verified."
echo "[prepare] Done."
