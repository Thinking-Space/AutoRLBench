#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_SOURCE="${MODEL_DIR:-${ROOT}/staged-model/Qwen3.5-9B}"
test -f "${MODEL_SOURCE}/config.json"
mkdir -p "${ROOT}/staged-model"
TARGET="${ROOT}/staged-model/Qwen3.5-9B"
if [[ ! -e "${TARGET}" ]]; then
  ln -s "$(readlink -f "${MODEL_SOURCE}")" "${TARGET}"
fi
test "$(readlink -f "${TARGET}")" = "$(readlink -f "${MODEL_SOURCE}")"
export JOB_ID="${JOB_ID:-cc-search-r1-$(date -u +%Y%m%dT%H%M%SZ)}"
exec timeout --signal=TERM --kill-after=60s "${RUN_TIMEOUT_SECONDS:-3600}" \
  bash "${ROOT}/code/entry.sh"
