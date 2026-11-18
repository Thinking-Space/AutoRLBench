#!/bin/bash
# MiniMax-M2.5 8L: 8GPU SFT 训练 (EP=8, TP=1)
# 裁剪到 8 层, 256 experts, ~60B total / ~11B active per forward
#
# SFT 模式：纯监督微调，不启动 sglang 推理引擎。
# 数据格式：JSONL，每行 {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
#
# 数据：autocuda_sft_glm5 (9149 samples, CUDA 优化任务, avg ~17k tokens, max ~55k tokens)
# 显存：8L 模型极小，瓶颈在长序列注意力; recompute_activations + FA3 保证可行
#
# Key parameters (override via env vars):
#   NUM_EPOCH            default 3
#   SFT_BATCH_SIZE       default 8
#   MAX_TOKENS_PER_GPU   default 65536
#   SFT_DATA             default data/sft/autocuda_sft_glm5.jsonl
#   SFT_LR               default 5e-6
#   TENSORBOARD_DIR      default exp_log/tensorboard/minimax_m2_8L_sft
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

export MODEL_PATH=${BASE_DIR}/models/MiniMax-M2.5-8L
export MEGATRON_MODEL_PATH=${BASE_DIR}/checkpoints/minimax-m2-8L-megatron
export CKPT_PATH=${BASE_DIR}/checkpoints
export SGLANG_PATH=${WORK_DIR}/sglang
export PYTHONPATH=.:Megatron-LM/:mbridge:${SGLANG_PATH}/python

mkdir -p /root/.cache/huggingface/modules/transformers_modules/
cp ${MODEL_PATH}/configuration_minimax_m2.py \
   ${MODEL_PATH}/modeling_minimax_m2.py \
   /root/.cache/huggingface/modules/transformers_modules/

TRAIN_FILE="${SFT_DATA:-${BASE_DIR}/data/sft/autocuda_sft_glm5.jsonl}"

export ARG_MODE=true_on_policy
export WANDB_API_KEY=i_dont_have_wandb_key
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${BASE_DIR}/exp_log/tensorboard/minimax_m2_8L_sft}

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

SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${SFT_BATCH_SIZE}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-65536}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-2048}
SFT_LR=${SFT_LR:-5e-6}

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="minimax_m2_8L_sft_${RUN_TAG}"

CKPT_ARGS=(
  --hf-checkpoint $MODEL_PATH
  --load $MEGATRON_MODEL_PATH
  --save ${CKPT_PATH}/${EXP_NAME}/
  --save-interval 1000
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
  --expert-model-parallel-size 8
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
)

WANDB_ARGS=(--use-tensorboard)

mkdir -p ${CKPT_PATH}/${EXP_NAME}/dump_details
MISC_ARGS=(
  --dump-details ${CKPT_PATH}/${EXP_NAME}/dump_details
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --moe-token-dispatcher-type alltoall
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\":  \".:./Megatron-LM:./mbridge:${SGLANG_PATH}/python:/root/.cache/huggingface/modules\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

bash scripts/setup/ray_start.sh

echo "============================================================"
echo "MiniMax-M2.5 8L SFT: ${NUM_EPOCH} epochs, EP=8, EXP=${EXP_NAME}"
echo "TRAIN_FILE: ${TRAIN_FILE}"
echo "lr=${SFT_LR}, batch_size=${SFT_BATCH_SIZE}, max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"
echo "recompute_activations=ON, attention_backend=flash(FA3)"
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
echo "MiniMax-M2.5 8L SFT done! EXP=${EXP_NAME}"
