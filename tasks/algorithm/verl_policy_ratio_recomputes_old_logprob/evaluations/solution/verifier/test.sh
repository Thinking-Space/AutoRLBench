#!/usr/bin/env bash
# Verifier scripts must retain LF endings for Linux containers.
set -uo pipefail

verifier="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
# The task ships PYTHONPATH=/workspace for training; strip it here so a planted
# workspace module cannot shadow the verifier's imports.
unset PYTHONPATH
exec python3 "$verifier/tests/grade_submission.py" "$@"
