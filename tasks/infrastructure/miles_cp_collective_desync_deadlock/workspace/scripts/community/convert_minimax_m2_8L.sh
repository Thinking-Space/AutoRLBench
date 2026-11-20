#!/bin/bash
# 转换 MiniMax-M2.5 8L HF (FP8) → Megatron checkpoint (BF16)
set -ex

cd /user/luozhenghang/0226_miles/miles

source scripts/setup/env.sh
export PYTHONPATH=./Megatron-LM/:./mbridge:.
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export CUDA_DEVICE_MAX_CONNECTIONS=1

BASE_DIR="$(cd "${WORK_DIR}/.." && pwd)"
HF_MODEL=${BASE_DIR}/models/MiniMax-M2.5-8L
SAVE_PATH=${BASE_DIR}/checkpoints/minimax-m2-8L-megatron

# 复制 modeling 文件到 HuggingFace cache
mkdir -p /root/.cache/huggingface/modules/transformers_modules/
cp ${HF_MODEL}/configuration_minimax_m2.py \
   ${HF_MODEL}/modeling_minimax_m2.py \
   /root/.cache/huggingface/modules/transformers_modules/

echo ">>> Converting MiniMax-M2.5 8L: ${HF_MODEL} → ${SAVE_PATH}"
echo ">>> mbridge location: $(python3 -c 'import mbridge; print(mbridge.__file__)')"
echo ">>> minimax_m2 bridge: $(python3 -c 'from mbridge.models.minimax_m2 import MiniMaxM2Bridge; import inspect; print(inspect.getfile(MiniMaxM2Bridge))')"

rm -rf ${SAVE_PATH}

torchrun --nproc-per-node 1 tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint ${HF_MODEL} \
    --save ${SAVE_PATH} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --disable-bias-linear \
    --num-layers 8 \
    --hidden-size 3072 \
    --ffn-hidden-size 1536 \
    --num-attention-heads 48 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 128 \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --norm-epsilon 1e-6 \
    --rotary-percent 0.5 \
    --rotary-base 5000000 \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --no-masked-softmax-fusion \
    --vocab-size 200064 \
    --make-vocab-size-divisible-by 64 \
    --qk-layernorm \
    --full-dim-qk-layernorm \
    --num-experts 256 \
    --moe-ffn-hidden-size 1536 \
    --moe-router-topk 8 \
    --moe-router-pre-softmax \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-load-balancing-type none \
    --moe-aux-loss-coeff 0 \
    --moe-router-bias-update-rate 0 \
    --moe-grouped-gemm \
    --moe-token-drop-policy probs \
    --moe-router-dtype fp32 \
    --bf16

echo "转换完成: ${SAVE_PATH}"
