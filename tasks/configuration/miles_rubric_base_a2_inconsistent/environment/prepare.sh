#!/usr/bin/env bash
# miles_rubric_base_a2_inconsistent — prepare stage.
# CPU-only task; no model weights needed.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
echo "[prepare] miles_rubric_base_a2_inconsistent"

pip install --quiet pyarrow==17.0.0 --target "${AUTORLBENCH_PREPARED}/pysite" 2>/dev/null || true

for f in gen_rubric_tigan10k.py data/answers_q359b.parquet data/answers_moebase.parquet data/rubric.jsonl data/train_mismatched.parquet; do
  [ -f "${AUTORLBENCH_WORKSPACE}/${f}" ] && [ -s "${AUTORLBENCH_WORKSPACE}/${f}" ] && echo "[prepare]   OK: ${f}" || { echo "[prepare] ERROR: ${f}" >&2; exit 1; }
done
echo "[prepare] Done."
