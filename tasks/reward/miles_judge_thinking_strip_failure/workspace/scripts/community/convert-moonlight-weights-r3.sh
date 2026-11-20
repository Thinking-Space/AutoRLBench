#!/bin/bash
# Moonlight-16B-A3B 权重转换脚本 - R3 配置 (TP=2, CP=2, EP=8)

set -x
source scripts/setup/env.sh
bash scripts/setup/setup.sh

export MODEL_PATH=/user/luozhenghang/1029_miles_miniCPM8b/models/Moonlight-16B-A3B
export SAVE_PATH=/user/luozhenghang/1029_miles_miniCPM8b/checkpoints/Moonlight-16B-A3B-megatron-r3
export PYTHONPATH=.:Megatron-LM/:../mbridge

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/moonlight.sh"

# R3 需要 TP/CP 并行，必须设置此环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Converting Moonlight-16B-A3B weights for R3 config (TP=2, EP=8)"
echo "From: $MODEL_PATH"
echo "To: $SAVE_PATH"

mkdir -p $SAVE_PATH

# R3 配置: TP=2, CP=2, EP=8
torchrun --nproc_per_node=8 \
    tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint $MODEL_PATH \
    --save $SAVE_PATH \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 8 \
    --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --context-parallel-size 2 \
    ${MODEL_ARGS[@]} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --moe-enable-deepep \
    --moe-token-dispatcher-type flex

echo "R3 权重转换完成！保存到: $SAVE_PATH"
