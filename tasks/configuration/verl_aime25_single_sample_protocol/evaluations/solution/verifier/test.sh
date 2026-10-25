#!/usr/bin/env bash
set -euo pipefail

: "${AUTORLBENCH_RESULTS:?AUTORLBENCH_RESULTS is required}"
: "${AUTORLBENCH_ORACLE:?AUTORLBENCH_ORACLE is required}"
: "${AUTORLBENCH_WORKSPACE:?AUTORLBENCH_WORKSPACE is required}"

PROTO="$AUTORLBENCH_RESULTS/solution_protocol.json"

if [ ! -f "$PROTO" ]; then
    echo "[FAIL] rerun did not produce solution_protocol.json" >&2
    echo "       expected: $PROTO" >&2
    exit 1
fi

python3 - "$PROTO" <<'PY'
import json
import sys
from pathlib import Path

p = json.loads(
    Path(sys.argv[1]).read_text(encoding="utf-8")
)

assert p["task"] == "aime25", p
assert p["benchmark"].lower() == "aime25", p
assert p["questions"] == 30, p
assert p["samples_per_question"] == 32, p
assert p["reducer"] == "mean", p
assert abs(p["temperature"] - 1.0) < 1e-12, p
assert abs(p["top_p"] - 1.0) < 1e-12, p
assert p["expected_generations"] == 960, p

print("[PASS] repaired AIME25 protocol is active")
PY

EVID="$AUTORLBENCH_WORKSPACE/autorlbench_case/original_eval"
MANIFEST="$AUTORLBENCH_ORACLE/original_eval.sha256"

if [ ! -f "$MANIFEST" ]; then
    echo "[FAIL] missing trusted oracle evidence manifest:" >&2
    echo "       $MANIFEST" >&2
    exit 1
fi

if [ ! -d "$EVID" ]; then
    echo "[FAIL] missing preserved original evidence:" >&2
    echo "       $EVID" >&2
    exit 1
fi

(
  cd "$EVID"
  sha256sum -c "$MANIFEST"
)

echo "[PASS] preserved original evaluation evidence is unchanged"
