#!/usr/bin/env bash
set -euo pipefail
RESULT_DIR="${1:?usage: test.sh <dir>}"
test -f "$RESULT_DIR/diagnosis.md" || { echo "missing diagnosis.md"; exit 1; }
echo "diagnosis.md present"
