#!/usr/bin/env bash
# miles_rubric_build_premature — prepare stage.
# CPU-only task; no model weights needed.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
echo "[prepare] miles_rubric_build_premature"

# Install pyarrow for parquet inspection
pip install --quiet pyarrow==17.0.0 --target "${AUTORLBENCH_PREPARED}/pysite" 2>/dev/null || true

for f in gen_rubric_tigan10k.py data/rubric_partial.jsonl data/rubric_complete_expected.jsonl data/train_premature.parquet; do
  [ -f "${AUTORLBENCH_WORKSPACE}/${f}" ] && [ -s "${AUTORLBENCH_WORKSPACE}/${f}" ] && echo "[prepare]   OK: ${f}" || { echo "[prepare] ERROR: ${f}" >&2; exit 1; }
done
echo "[prepare] Done."
