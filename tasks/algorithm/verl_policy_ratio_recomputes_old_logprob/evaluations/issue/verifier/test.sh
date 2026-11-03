#!/usr/bin/env bash
set -euo pipefail

verifier="${AUTORLBENCH_VERIFIER:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export PYTHONDONTWRITEBYTECODE=1
# The task ships PYTHONPATH=/workspace for training; strip it here so a planted
# workspace module cannot shadow the judge's imports.
unset PYTHONPATH
# Standalone evaluate has no agent phase. Accept a caller-supplied workspace
# diagnosis when Harbor has not already written the normal results file.
if [[ ! -f "${AUTORLBENCH_RESULTS}/agent_answer.md" && -f "${AUTORLBENCH_WORKSPACE}/agent_answer.md" ]]; then
  set -- --answer "${AUTORLBENCH_WORKSPACE}/agent_answer.md" "$@"
fi
exec python3 "$verifier/judge.py" "$@"
