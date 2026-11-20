#!/usr/bin/env bash
# HF -> Megatron torch_dist conversion for MiniCPM5-2.6B

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/minicpm5-26b.sh"

export PYTHONPATH="${REPO_ROOT}/Megatron-LM/:${REPO_ROOT}/mbridge:${REPO_ROOT}:${PYTHONPATH:-}"
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
export CUDA_DEVICE_MAX_CONNECTIONS=1

HF_CHECKPOINT=${HF_CHECKPOINT:-/user/yanhui/share_user_long/lijiayi/2.6b/MiniCPM5-2.6B-0426_job_327123_step_24000_fusion_think}
SAVE=${SAVE:-/user/panyinxu/ckpts/MiniCPM5-2.6B-0426_job_327123_step_24000_fusion_think_tp1_torch_dist}

# Default to TP=1 conversion; override by env if needed.
NUM_GPUS=${NUM_GPUS:-1}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}

torchrun --nproc-per-node "${NUM_GPUS}" "${REPO_ROOT}/tools/convert_hf_to_torch_dist.py" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --save "${SAVE}" \
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
    --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}" \
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE}" \
    --bf16 \
    "${MODEL_ARGS[@]}"
