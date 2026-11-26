#!/usr/bin/env bash
set -uo pipefail

verifier="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$verifier/tests/grade_submission.py" "$@"
