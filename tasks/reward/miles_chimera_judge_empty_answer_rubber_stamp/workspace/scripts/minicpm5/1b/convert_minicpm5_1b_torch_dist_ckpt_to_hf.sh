#!/usr/bin/env bash
# Convert MiniCPM5-1B torch_dist checkpoint to HuggingFace format.

set -euo pipefail
set -x
export PYTHONPATH=".:${PYTHONPATH:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/minicpm5-1b.sh"

INPUT_DIR=${1:-/user/panyinxu/ckpts/MiniCPM5-1B-SFT_torch_dist}
OUTPUT_DIR=${2:-${INPUT_DIR}_hf}
ORIGIN_HF_DIR=${ORIGIN_HF_DIR:-/user/panyinxu/models/MiniCPM5-1B-SFT}

VOCAB_SIZE=""
for ((i = 0; i < ${#MODEL_ARGS[@]}; i++)); do
    if [[ "${MODEL_ARGS[$i]}" == "--vocab-size" ]]; then
        VOCAB_SIZE="${MODEL_ARGS[$((i + 1))]}"
        break
    fi
done

if [[ -z "${VOCAB_SIZE}" ]]; then
    echo "Failed to find --vocab-size in ${SCRIPT_DIR}/minicpm5-1b.sh" >&2
    exit 1
fi

python3 "${REPO_ROOT}/tools/convert_torch_dist_to_hf.py" \
    --input-dir "${INPUT_DIR}" \
    --origin-hf-dir "${ORIGIN_HF_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --vocab-size "${VOCAB_SIZE}" \
    --force
