#!/usr/bin/env bash
# HF → Megatron torch_dist，与 scripts/minicpm5/16a3b/minicpm5_moe_16a3b_rope.sh
# 和 scripts/minicpm5/16a3b 下的训练脚本配置对齐。
#
# 默认 checkpoint：job_362482.step_47000。可覆盖：
#   HF_CHECKPOINT=... SAVE=... torchrun ... 见下文

set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/minicpm5_moe_16a3b_rope.sh"

export PYTHONPATH="${REPO_ROOT}/Megatron-LM/:${REPO_ROOT}/mbridge:${REPO_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
export CUDA_DEVICE_MAX_CONNECTIONS=1

BASE_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
HF_CHECKPOINT=${HF_CHECKPOINT:-/user/panyinxu/models/job_362482_step_47000}
SAVE=${SAVE:-/user/panyinxu/ckpts/minicpm5_362482_step47000}

torchrun --nproc-per-node 1 "${REPO_ROOT}/tools/convert_hf_to_torch_dist.py" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --save "${SAVE}" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --bf16 \
    "${MODEL_ARGS[@]}"
