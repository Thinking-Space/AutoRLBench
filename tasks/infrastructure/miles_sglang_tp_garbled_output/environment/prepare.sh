#!/usr/bin/env bash
# miles_sglang_tp_garbled_output — prepare stage.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
echo "[prepare] miles_sglang_tp_garbled_output"
for f in scripts/run_training.sh outputs/driver.log; do
  [ -f "${AUTORLBENCH_WORKSPACE}/${f}" ] && [ -s "${AUTORLBENCH_WORKSPACE}/${f}" ] && echo "[prepare]   OK: ${f}" || { echo "[prepare] ERROR: ${f}" >&2; exit 1; }
done
echo "[prepare] Done."
