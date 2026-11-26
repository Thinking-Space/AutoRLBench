#!/usr/bin/env bash
# verifier/test.sh —— 素材阶段占位：确认诊断报告存在。
set -euo pipefail
RESULT_DIR="${1:?usage: test.sh <agent-result-dir>}"
test -f "$RESULT_DIR/diagnosis.md" || { echo "missing diagnosis.md"; exit 1; }
echo "diagnosis.md present"