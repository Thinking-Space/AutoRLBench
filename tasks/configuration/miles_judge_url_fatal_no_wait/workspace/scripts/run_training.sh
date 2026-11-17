#!/usr/bin/env bash
# Training launch script (defective version)
# BUG: judge URL detection fatals immediately if file absent instead of waiting.

set -ex

# ---- 判官 URL 动态发现 (DEFECTIVE VERSION) ----
# BUG: checks once and exits immediately if file absent.
# The judge job was still Queued and had not yet written its URL file.
_JUDGE_URL_FILE=$(ls /scratch/judge_ips/*/*.url 2>/dev/null | head -1)
if [ -z "${_JUDGE_URL_FILE}" ] || [ ! -f "${_JUDGE_URL_FILE}" ]; then
  echo "[FATAL] 找不到判官 URL 文件" >&2
  exit 1
fi
export RUBRIC_JUDGE_URL=$(cat "${_JUDGE_URL_FILE}")
echo "[judge] using ${RUBRIC_JUDGE_URL}"

# Correct version would poll in a loop:
# for _i in $(seq 1 180); do
#   _JUDGE_URL_FILE=$(ls /scratch/judge_ips/*/*.url 2>/dev/null | head -1)
#   if [ -n "${_JUDGE_URL_FILE}" ] && [ -s "${_JUDGE_URL_FILE}" ]; then
#     export RUBRIC_JUDGE_URL=$(cat "${_JUDGE_URL_FILE}")
#     break
#   fi
#   sleep 30
# done

python3 -m miles.train --num-rollout 200 "$@"
