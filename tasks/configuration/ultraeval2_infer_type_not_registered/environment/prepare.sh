#!/usr/bin/env bash
# ultraeval2_infer_type_not_registered — prepare stage.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
echo "[prepare] ultraeval2_infer_type_not_registered"
for f in scripts/submit_eval.sh outputs/scheduler_state.json specs/eval_think.json; do
  [ -f "${AUTORLBENCH_WORKSPACE}/${f}" ] && [ -s "${AUTORLBENCH_WORKSPACE}/${f}" ] && echo "[prepare]   OK: ${f}" || { echo "[prepare] ERROR: ${f}" >&2; exit 1; }
done
echo "[prepare] Done."
