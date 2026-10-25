#!/usr/bin/env bash
set -euo pipefail
: "${PTDBENCH_RESULTS:?PTDBENCH_RESULTS must be set}"
: "${PTDBENCH_ORACLE:?PTDBENCH_ORACLE must be set}"
: "${PTDBENCH_VERIFIER:?PTDBENCH_VERIFIER must be set}"
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$PTDBENCH_VERIFIER/judge.py" "$@"
