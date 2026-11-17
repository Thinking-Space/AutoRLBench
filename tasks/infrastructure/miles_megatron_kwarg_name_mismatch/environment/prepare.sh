#!/usr/bin/env bash
# miles_megatron_kwarg_name_mismatch — prepare stage.
#
# Runs in the miles-0.1.1 shared image. This task is a pure Python
# version-mismatch defect: no model weights or datasets need downloading.
# The workspace already contains a minimal stub of the defective code that
# reproduces the TypeError without launching a full training run.
#
# prepare.sh verifies the workspace files are present and non-empty.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"

echo "[prepare] miles_megatron_kwarg_name_mismatch"
echo "[prepare] Verifying workspace files..."

for f in \
  miles_plugins/models/hf_attention.py \
  megatron_stub/transformer_layer.py \
  reproduce.py \
  outputs/worker_stderr.log \
  scripts/run_training.sh
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
