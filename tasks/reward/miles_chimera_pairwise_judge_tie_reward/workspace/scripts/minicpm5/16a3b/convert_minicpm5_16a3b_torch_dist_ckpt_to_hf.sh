#!/bin/bash
# 将 MiniCPM5 MoE 16A3B 128k 的 torch_dist 训练 checkpoint 转为 HuggingFace 格式

set -euo pipefail
set -x
export PYTHONPATH=".:${PYTHONPATH:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/minicpm5_moe_16a3b_rope.sh"

INPUT_DIR=${1:-/para_hdd/user/panyinxu/projects/miles_minicpm5_rl_scaling-1847/399081/checkpoints/minicpm5_16a3b_scaling_20260525_155723/iter_0000149}
OUTPUT_DIR=${2:-/user/panyinxu/ckpts/minicpm5_16a3b_128k_swe_rl_r2egym_step150}
ORIGIN_HF_DIR=${ORIGIN_HF_DIR:-/user/panyinxu/models/job_362482_step_47000}

VOCAB_SIZE=""
for ((i = 0; i < ${#MODEL_ARGS[@]}; i++)); do
    if [[ "${MODEL_ARGS[$i]}" == "--vocab-size" ]]; then
        VOCAB_SIZE="${MODEL_ARGS[$((i + 1))]}"
        break
    fi
done

if [[ -z "${VOCAB_SIZE}" ]]; then
    echo "Failed to find --vocab-size in ${SCRIPT_DIR}/minicpm5_moe_16a3b_rope.sh" >&2
    exit 1
fi

python3 "${REPO_ROOT}/tools/convert_torch_dist_to_hf.py" \
    --input-dir "${INPUT_DIR}" \
    --origin-hf-dir "${ORIGIN_HF_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --vocab-size "${VOCAB_SIZE}" \
    --force
