#!/bin/bash
# MiniMax-M2.5 Full: 64GPU SFT 训练 (EP=32, TP=1, DP_expert=2)
# 62 层全参数, 256 experts (topk=8), ~228B total / ~11B active per forward
#
# SFT 模式：纯监督微调，不启动 sglang 推理引擎。
# 数据格式：JSONL，每行 {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
#
# 硬件要求：8 节点 × 8 GPU H100 80GB = 64 GPU
#
# ==================== 严格显存估算 ====================
# 并行策略: EP=32, TP=1, PP=1 → DP_dense=64, DP_expert=2
# precision_aware_optimizer: master_int16(2B) + exp_avg_fp32(4B) + exp_avg_sq_fp32(4B) = 10B/param
#
# Expert params/GPU (EP=32: 8 experts/layer × 62 layers):
#   weights(bf16): 7.02B × 2 = 14.04 GB
#   grad_buffer(bf16, full): 7.02B × 2 = 14.04 GB
#   optimizer(sharded by DP_expert=2): 7.02B × 10 / 2 = 35.10 GB
#   subtotal: 63.18 GB
#
# Non-expert params/GPU (4.01B, replicated):
#   weights(bf16): 4.01B × 2 = 8.02 GB
#   grad_buffer(bf16, full): 4.01B × 2 = 8.02 GB
#   optimizer(sharded by DP_dense=64): 4.01B × 10 / 64 = 0.63 GB
#   subtotal: 16.67 GB
#
# CUDA context + framework overhead: ~5 GB
# 总静态: ~84.85 GB → 对于长序列极其紧张!
#
# 本脚本用于 **测试数据** (短序列 ~1024 tokens):
#   activations: ~0.4 GB → 总计 ~85 GB → 勉强可行
#   logit_chunk=256: 256 × 200064 × 4B = 196 MB → 安全
#
# 注意: 正式长序列数据请使用 run-minimax-m2.5-64gpu-full-sft-autocuda.sh
#       该脚本使用 bf16 optimizer states 来腾出显存
# =======================================================
#
# Key parameters (override via env vars):
#   NUM_EPOCH            default 3
#   SFT_BATCH_SIZE       default 64
#   MAX_TOKENS_PER_GPU   default 2048
#   LOG_PROBS_CHUNK_SIZE default 256 (从8L经验降低, 防止OOM)
#   SFT_DATA             default data/sft/sft_test.jsonl
#   SFT_LR               default 5e-6
#   TENSORBOARD_DIR      default exp_log/tensorboard/minimax_m2.5_full_sft
set -x
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source scripts/setup/env.sh
bash scripts/setup/setup.sh
export SLIME_SCRIPT_NUM_GPUS=$GPUS_PER_NODE

# ============================================================
# 路径配置
# ============================================================
BASE_DIR="$(cd "${WORK_DIR}/.." && pwd)"

export MODEL_PATH=${BASE_DIR}/models/MiniMax-M2.5
MEGATRON_CKPT_DIR=${BASE_DIR}/checkpoints/minimax-m2.5-full-megatron
if [ -f "${MEGATRON_CKPT_DIR}/latest_checkpointed_iteration.txt" ]; then
  export MEGATRON_MODEL_PATH=${MEGATRON_CKPT_DIR}
  echo "Found Megatron checkpoint at ${MEGATRON_CKPT_DIR}, loading directly."
else
  echo "============================================================"
  echo "ERROR: Megatron checkpoint not found at ${MEGATRON_CKPT_DIR}"
  echo "Please run the conversion script first:"
  echo "  bash scripts/community/convert_minimax_m2_full.sh"
  echo "============================================================"
  exit 1
fi
export CKPT_PATH=${BASE_DIR}/checkpoints
export SGLANG_PATH=${WORK_DIR}/sglang
export PYTHONPATH=.:Megatron-LM/:mbridge:${SGLANG_PATH}/python

mkdir -p /root/.cache/huggingface/modules/transformers_modules/
cp ${MODEL_PATH}/configuration_minimax_m2.py \
   ${MODEL_PATH}/modeling_minimax_m2.py \
   /root/.cache/huggingface/modules/transformers_modules/

TRAIN_FILE="${SFT_DATA:-${BASE_DIR}/data/sft/sft_test.jsonl}"

export ARG_MODE=true_on_policy
export WANDB_API_KEY=i_dont_have_wandb_key
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${BASE_DIR}/exp_log/tensorboard/minimax_m2.5_full_sft}

# ============================================================
# 可配置参数
# ============================================================
NUM_EPOCH=${NUM_EPOCH:-3}

if [ "${SKIP_KILL:-0}" != "1" ]; then
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -f "python3 train.py" || true
  sleep 3
  pkill -9 ray || true
  pkill -f "python3 train.py" || true
  pkill -9 redis || true
else
  echo "SKIP_KILL=1: skip pkill/ray stop cleanup"
fi

set -ex
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$([[ "$NVLINK_COUNT" -gt 0 ]] && echo 1 || echo 0)

export NLAYERS=62

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/minimax_m2.sh"

MODEL_ARGS_FAST=()
skip_next=0
for a in "${MODEL_ARGS[@]}"; do
  if [ "$skip_next" -eq 1 ]; then
    skip_next=0
    continue
  fi
  case "$a" in
    --no-masked-softmax-fusion) continue ;;
    --moe-router-dtype) skip_next=1; continue ;;
    --moe-token-dispatcher-type) skip_next=1; continue ;;
    *) MODEL_ARGS_FAST+=("$a") ;;
  esac
done
MODEL_ARGS_FAST+=(
  --bf16
  --grad-reduce-in-bf16
  --main-grads-dtype bf16
  --moe-router-dtype fp32
)

SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-64}
GLOBAL_BATCH_SIZE=${SFT_BATCH_SIZE}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-2048}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-256}
SFT_LR=${SFT_LR:-5e-6}

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="minimax_m2.5_full_sft_${RUN_TAG}"

CKPT_ARGS=(
  --hf-checkpoint $MODEL_PATH
  --load $MEGATRON_MODEL_PATH
  --save ${CKPT_PATH}/${EXP_NAME}/
  --save-interval 500
)

SFT_ARGS=(
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data $TRAIN_FILE
  --input-key messages
  --rollout-shuffle
  --num-epoch ${NUM_EPOCH}
  --rollout-batch-size ${SFT_BATCH_SIZE}
  --global-batch-size ${GLOBAL_BATCH_SIZE}
  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

PERF_ARGS=(
  --train-backend megatron
  --tensor-model-parallel-size 1
  --context-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 32
  --expert-tensor-parallel-size 1
  --attention-backend flash
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
  --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
  --recompute-activations
  --train-memory-margin-bytes 0
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr ${SFT_LR}
  --lr-decay-style cosine
  --min-lr 1e-7
  --lr-warmup-fraction 0.05
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --use-precision-aware-optimizer
  --use-distributed-optimizer
)

WANDB_ARGS=(--use-tensorboard)

mkdir -p ${CKPT_PATH}/${EXP_NAME}/dump_details
MISC_ARGS=(
  --dump-details ${CKPT_PATH}/${EXP_NAME}/dump_details
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --moe-token-dispatcher-type alltoall
)

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=eth0

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\":  \".:./Megatron-LM:./mbridge:${SGLANG_PATH}/python:/root/.cache/huggingface/modules\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_IB_DISABLE\": \"0\",
    \"NCCL_NET_GDR_LEVEL\": \"2\"
  }
}"

bash scripts/setup/ray_start.sh

echo "============================================================"
echo "MiniMax-M2.5 Full SFT [TEST DATA]: ${NUM_EPOCH} epochs, EP=32, TP=1, 64 GPUs"
echo "DP_expert=2, distributed_optimizer=ON, recompute_activations=ON"
echo "log_probs_chunk_size=${LOG_PROBS_CHUNK_SIZE} (reduced from 8L OOM experience)"
echo "TRAIN_FILE: ${TRAIN_FILE}"
echo "lr=${SFT_LR}, batch_size=${SFT_BATCH_SIZE}, max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"
echo "attention_backend=flash(FA3)"
echo "BASE_DIR: ${BASE_DIR}"
echo "============================================================"

ray job submit --address="http://${MASTER_ADDR}:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes ${WORLD_SIZE} \
  --actor-num-gpus-per-node ${GPUS_PER_NODE} \
  --colocate \
  "${MODEL_ARGS_FAST[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}" "$@"

bash $WORK_DIR/scripts/setup/ray_end.sh
echo "MiniMax-M2.5 Full SFT done! EXP=${EXP_NAME}"
