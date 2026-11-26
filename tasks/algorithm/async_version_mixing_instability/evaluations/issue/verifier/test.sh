#!/usr/bin/env bash
# verifier/test.sh —— 素材阶段占位测试：确认 agent 的诊断报告存在且 oracle 脚本可运行。
# 正式 task 时按 reference.md / verifier 规范替换。
set -euo pipefail

RESULT_DIR="${1:?usage: test.sh <agent-result-dir>}"
test -f "$RESULT_DIR/diagnosis.md" || { echo "missing diagnosis.md"; exit 1; }
echo "diagnosis.md present"
