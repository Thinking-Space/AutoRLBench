#!/usr/bin/env bash
# miles_cctl_stop_needs_yes — prepare stage.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
echo "[prepare] miles_cctl_stop_needs_yes"
for f in scripts/stop_jobs.sh tools/cctl.sh outputs/cctl_stop_response.json; do
  [ -f "${AUTORLBENCH_WORKSPACE}/${f}" ] && [ -s "${AUTORLBENCH_WORKSPACE}/${f}" ] && echo "[prepare]   OK: ${f}" || { echo "[prepare] ERROR: ${f}" >&2; exit 1; }
done
echo "[prepare] Done."
