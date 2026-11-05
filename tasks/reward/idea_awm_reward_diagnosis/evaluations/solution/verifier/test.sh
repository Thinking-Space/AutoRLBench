#!/usr/bin/env bash
set -euo pipefail

: "${AUTORLBENCH_WORKSPACE:?AUTORLBENCH_WORKSPACE is required}"
: "${AUTORLBENCH_RESULTS:?AUTORLBENCH_RESULTS is required}"

verifier="${AUTORLBENCH_VERIFIER:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$verifier/tests/grade_submission.py" "$@"
