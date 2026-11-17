#!/usr/bin/env bash
# miles_judge_url_fatal_no_wait — prepare stage.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
echo "[prepare] miles_judge_url_fatal_no_wait"
for f in scripts/run_training.sh outputs/driver.log scratch/judge_ips; do
  [ -e "${AUTORLBENCH_WORKSPACE}/${f}" ] && echo "[prepare]   OK: ${f}" || { echo "[prepare] ERROR: ${f}" >&2; exit 1; }
done
echo "[prepare] Done."
