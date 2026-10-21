#!/usr/bin/env bash
set -euo pipefail
: "${PTDBENCH_WORKSPACE:?}" "${PTDBENCH_VERIFIER:?}" "${PTDBENCH_RESULTS:?}"
export W=$PTDBENCH_WORKSPACE
export HOME="$PTDBENCH_RESULTS/home" TMPDIR="$PTDBENCH_RESULTS/tmp"
mkdir -p "$HOME" "$TMPDIR"
variant=$(cat "$W/variant.txt")
export PYTHONPATH="$W/src/$variant:$W/scripts:$PTDBENCH_VERIFIER/tests"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRAINING_RUNTIME_HOOK=1
exec "${PTDBENCH_PYTHON:-/usr/bin/python3.12}" "$PTDBENCH_VERIFIER/tests/verify.py"
