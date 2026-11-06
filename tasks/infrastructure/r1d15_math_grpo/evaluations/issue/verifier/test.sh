#!/usr/bin/env bash
set -euo pipefail

verifier="${AUTORLBENCH_VERIFIER:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export PYTHONDONTWRITEBYTECODE=1
python3 - "${AUTORLBENCH_RESULTS:-/results}/agent_answer.md" <<'PY_SUBMISSION'
import json
from pathlib import Path
import sys

answer = Path(sys.argv[1])
if answer.is_symlink() or not answer.is_file():
    result = {
        "passed": False,
        "score": None,
        "metrics": {"status": "invalid_submission", "error": "agent_answer.md must be a regular file"},
    }
    (answer.parent / "result.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
    raise SystemExit(2)
PY_SUBMISSION
exec python3 "$verifier/judge.py" "$@"
