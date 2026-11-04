#!/usr/bin/env bash
# Verifier scripts must retain LF endings for Linux containers.
set -uo pipefail

verifier="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec python3 -I -S "$verifier/tests/grade_submission.py" "$@"
