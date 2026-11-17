#!/usr/bin/env bash
# miles_hf_convert_silent_skip — prepare stage.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
echo "[prepare] miles_hf_convert_silent_skip"
for f in scripts/run_training.sh scripts/ray_end.sh outputs/driver.log; do
  [ -f "${AUTORLBENCH_WORKSPACE}/${f}" ] && [ -s "${AUTORLBENCH_WORKSPACE}/${f}" ] && echo "[prepare]   OK: ${f}" || { echo "[prepare] ERROR: ${f}" >&2; exit 1; }
done
echo "[prepare] Done."
