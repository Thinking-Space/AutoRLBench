#!/bin/bash
# MiniMax-M2.5 Full: 64GPU SFT 训练 — autocuda 正式数据 (PP=8 长序列方案)
# 62 层全参数, 256 experts (topk=8), ~228B total / ~11B active per forward
#
# SFT 模式：纯监督微调，不启动 sglang 推理引擎。
# 数据格式：JSONL，每行 {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
#
# 硬件要求：8 节点 × 8 GPU H100 80GB = 64 GPU
#
# ====================== 显存估算 (PP=8 方案) ======================
# 并行策略: TP=1, PP=8, EP=8, CP=8
#   每 pipeline stage: 64/8 = 8 GPUs
#   DP = 8, EP=8 → DP_expert=1, CP=8 → DP/CP=1
#   层数/stage: 8×7 + 6 (--decoder-last-pipeline-num-layers 6)
#   GBS=16, num_microbatches = 16/(DP/CP) = 16, pipeline 效率 = 16/23 ≈ 70%
#
# 优势: PP=8 把 62 层分 8 段, 每 GPU 只扛 8 层 (权重+梯度减半)
#        CP=8 把序列切 8 份, 55k tokens → 6875 tokens/GPU
#        显存余量充足 (~20 GB), 不会 OOM
#
# 每 Expert FFN (SwiGLU): 3 × 3072 × 1536 = 14.16M params
# Expert params/GPU (EP=8: 32 experts/layer × 8 layers = 3.625B, worst-case 8L stage):
#   weights(bf16):                        3.625B × 2       =  7.25 GB
#   grad_buffer(bf16):                    3.625B × 2       =  7.25 GB
#   optimizer(DP_expert=1, 不分片):
#     main_params(fp32):                  3.625B × 4       = 14.50 GB
#     exp_avg(bf16):                      3.625B × 2       =  7.25 GB
#     exp_avg_sq(bf16):                   3.625B × 2       =  7.25 GB
#   expert subtotal:                                         43.50 GB
#
# Non-expert params/GPU (worst-case 8L stage: 8L × 44.84M = 0.359B):
#   weights(bf16):                        0.359B × 2       =  0.72 GB
#   grad_buffer(bf16):                    0.359B × 2       =  0.72 GB
#   optimizer(sharded by DP=8):
#     main_params(fp32):                  0.359B × 4 / 8   =  0.18 GB
#     exp_avg(bf16):                      0.359B × 2 / 8   =  0.09 GB
#     exp_avg_sq(bf16):                   0.359B × 2 / 8   =  0.09 GB
#   non-expert subtotal:                                      1.80 GB
#
# CUDA context + NCCL buffers (PP+EP+CP) + TE + Ray: ~13 GB (实测)
# 总静态 (worst-case 8L stage): ~58.3 GB / 80 GB
#
# Activation (FULL recompute + FA3, with CP=8):
#   full recompute 只保留每层输入 hidden_state, 层内全部重算
#   checkpoint 输入: 8 × 8192 × 3072 × 2B            =  384 MB
#   backward 单层重算峰值 (attn+MoE 中间激活):       ~ 1.5 GB
#   logit_chunk=128: 128 × 200064 × 4B (last stage)  =   98 MB
#   activation 峰值合计:                              ~ 2.0 GB
#
# ========================= 最终汇总 =============================
# PyTorch 静态 (weights+grads+optimizer):               45.3 GB
# 非 PyTorch (NCCL+TE+Ray, PP+EP+CP 通信):            ~13.0 GB
# activation 峰值 (full recompute):                      2.0 GB
# 总峰值:                                              ~60.3 GB
# H100 VRAM:                                            80.0 GB
# 净余量:                                              ~19.7 GB
# =================================================================
#
# 数据: autocuda_sft_glm5 (9149 samples, 全量不截断不过滤)
#   avg ~17k tokens, max ~55k tokens → CP=8 后 per-GPU max ~7k tokens
#
# Key parameters (override via env vars):
#   NUM_EPOCH            default 3
#   SFT_BATCH_SIZE       default 16
#   MAX_TOKENS_PER_GPU   default 8192  (CP=8 后 per-GPU, 即总容量 ~65k tokens)
#   LOG_PROBS_CHUNK_SIZE default 128
#   SFT_DATA             default data/sft/autocuda_sft_glm5.jsonl
#   SFT_LR               default 5e-5
#   TENSORBOARD_DIR      default exp_log/tensorboard/minimax_m2.5_full_sft_autocuda
set -x
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY

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

# ============================================================
# 数据: 全量使用, 不截断不过滤 (CP=8 自动分摊长序列)
# ============================================================
TRAIN_FILE="${SFT_DATA:-${BASE_DIR}/data/sft/autocuda_sft_glm5.jsonl}"
if [ ! -f "${TRAIN_FILE}" ]; then
  echo "ERROR: Training data not found: ${TRAIN_FILE}"
  echo "Please convert the data first (see run-minimax-m2-8gpu-8L-sft.sh)"
  exit 1
fi

export ARG_MODE=true_on_policy
export WANDB_API_KEY=i_dont_have_wandb_key
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${BASE_DIR}/exp_log/tensorboard/minimax_m2.5_full_sft_autocuda}

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

SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-16}
GLOBAL_BATCH_SIZE=${SFT_BATCH_SIZE}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-128}
SFT_LR=${SFT_LR:-5e-5}

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="minimax_m2.5_full_sft_autocuda_${RUN_TAG}"

CKPT_ARGS=(
  --hf-checkpoint $MODEL_PATH
  --load $MEGATRON_MODEL_PATH
  --save ${CKPT_PATH}/${EXP_NAME}/
  --save-interval 50
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
  --pipeline-model-parallel-size 8
  --decoder-last-pipeline-num-layers 6
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --context-parallel-size 8
  --sequence-parallel
  --attention-backend flash
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
  --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --train-memory-margin-bytes 0
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr ${SFT_LR}
  --lr-decay-style cosine
  --min-lr 5e-6
  --lr-warmup-fraction 0.05
  --weight-decay 0.0
  --adam-beta1 0.9
  --adam-beta2 0.98
  --use-precision-aware-optimizer
  --use-distributed-optimizer
  --exp-avg-dtype bf16
  --exp-avg-sq-dtype bf16
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
    \"FORCE_PYTORCH_OPTIMIZER\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_IB_DISABLE\": \"0\",
    \"NCCL_NET_GDR_LEVEL\": \"2\"
  }
}"

bash scripts/setup/ray_start.sh

echo "============================================================"
echo "MiniMax-M2.5 Full SFT [AUTOCUDA DATA] PP=8 长序列方案"
echo "${NUM_EPOCH} epochs, 64 GPUs"
echo "TP=1, PP=8(8×7+6), EP=8, CP=8 → DP=8, DP_expert=1"
echo "distributed_optimizer=ON, FULL recompute=ON (uniform, every layer)"
echo "main_params=fp32(default), exp_avg=bf16, exp_avg_sq=bf16"
echo "log_probs_chunk_size=${LOG_PROBS_CHUNK_SIZE}"
echo "max_tokens_per_gpu=${MAX_TOKENS_PER_GPU} (with CP=8: effective ~${MAX_TOKENS_PER_GPU}×8 total)"
echo "TRAIN_FILE: ${TRAIN_FILE} (全量数据, 不截断不过滤)"
echo "lr=${SFT_LR}, batch_size=${SFT_BATCH_SIZE}"
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
echo "MiniMax-M2.5 Full SFT autocuda (PP=8) done! EXP=${EXP_NAME}"
