#!/usr/bin/env bash
set -euo pipefail

verifier="${AUTORLBENCH_VERIFIER:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$verifier/judge.py" "$@"
