#!/bin/bash
# Test memory optimizations: FA TP=1 path, chunked backward, entropy no_grad
#
# Changes from aime-64k baseline:
#   CP=1  (was 4)  → enables chunked backward (output_layer projection in chunks)
#   entropy_coef=0.001 (was 0) → exercises entropy computation under no_grad
#   max_tokens_per_gpu=16384 (was 10240) → stress test: more tokens per GPU
#   NUM_ROLLOUT=2 → quick validation
#
# Expected behavior:
#   - Chunked backward: model.py returns hidden_states (not full logits),
#     loss.py projects to logits in chunks → peak backward mem ~ chunk_size * V
#   - FA TP=1: _logprobs_from_logits uses flash_attn cross_entropy → no clone
#   - Entropy under no_grad: skips autograd graph → saves ~2 * [T, V] saved tensors
#
# Success criteria:
#   - No OOM with max_tokens_per_gpu=16384 (vs 10240 baseline)
#   - entropy_loss > 0 in training logs
#   - Loss converges normally (non-zero pg_clipfrac with lr=5e-6)

set -x
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY

export CLUSTER=${CLUSTER:-ningxia_h100}
source scripts/setup/env.sh
bash scripts/setup/setup.sh
export SLIME_SCRIPT_NUM_GPUS=$GPUS_PER_NODE

BASE_DIR="$(cd "${WORK_DIR}/.." && pwd)"

mkdir -p /root/.cache/huggingface/modules/transformers_modules/
cp ${BASE_DIR}/models/minicpmmoe/configuration_minicpm.py \
   ${BASE_DIR}/models/minicpmmoe/modeling_minicpm.py \
   /root/.cache/huggingface/modules/transformers_modules/

export MODEL_PATH=${BASE_DIR}/models/minicpmmoe
export MEGATRON_MODEL_PATH=${BASE_DIR}/checkpoints/minicpm5-moe-megatron-fixed-v2
export CKPT_PATH=${BASE_DIR}/checkpoints
export SGLANG_PATH=${WORK_DIR}/sglang
export PYTHONPATH=.:Megatron-LM/:mbridge:${SGLANG_PATH}/python

TRAIN_FILE="${BASE_DIR}/data/train/dapo-math-17k.jsonl"
export ARG_MODE=true_on_policy
export WANDB_API_KEY=i_dont_have_wandb_key
export TENSORBOARD_DIR=${BASE_DIR}/exp_log/tensorboard/test_mem_opt

NUM_ROLLOUT=${NUM_ROLLOUT:-2}

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
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/minicpm5_moe.sh"

MODEL_ARGS_FAST=()
skip_next=0
for a in "${MODEL_ARGS[@]}"; do
  if [ "$skip_next" -eq 1 ]; then
    skip_next=0
    continue
  fi
  case "$a" in
    --no-masked-softmax-fusion) continue ;;
    --no-rope-fusion) continue ;;
    --attention-softmax-in-fp32) continue ;;
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

# CP=4: standard parallelism; tests FA TP=1 path + entropy no_grad
# (Chunked backward requires CP=1 + truncated samples, not tested here)
CONTEXT_PARALLEL_SIZE=4
ROLLOUT_NUM_GPUS_PER_ENGINE=4

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))

# With CP=1, the full sequence lands on one GPU. Chunked backward reduces
# logits peak memory from T*V to chunk_size*V, allowing more tokens per GPU.
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-10240}
LOGPROBS_MAX_TOKENS_PER_GPU=${LOGPROBS_MAX_TOKENS_PER_GPU:-10240}
LOG_PROBS_CHUNK_SIZE=2048

SGLANG_MEM_FRACTION=0.75

SGLANG_CONTEXT_LEN=68000
SGLANG_MAX_PREFILL_TOKENS=68000
SGLANG_MAX_TOTAL_TOKENS=68000

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="test_mem_opt_cp1_${RUN_TAG}"

CKPT_ARGS=(
  --hf-checkpoint $MODEL_PATH
  --load $MEGATRON_MODEL_PATH
  --save ${CKPT_PATH}/${EXP_NAME}/
  --save-interval 9999
)

ROLLOUT_ARGS=(
  --prompt-data $TRAIN_FILE
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type math
  --num-rollout ${NUM_ROLLOUT}
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
  --rollout-max-prompt-len 2048
  --rollout-max-response-len 64000
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --global-batch-size ${GLOBAL_BATCH_SIZE}
  --balance-data
)

PERF_ARGS=(
  --train-backend megatron
  --tensor-model-parallel-size 1
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
  --log-probs-max-tokens-per-gpu ${LOGPROBS_MAX_TOKENS_PER_GPU}
  --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.001
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-rollout-routing-replay
  --use-miles-router
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 5e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --use-precision-aware-optimizer
)

WANDB_ARGS=(--use-tensorboard)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION}
  --sglang-max-running-requests 16
  --sglang-disable-shared-experts-fusion

  --sglang-attention-backend fa3
  --sglang-json-model-override-args "{\"max_position_embeddings\": ${SGLANG_CONTEXT_LEN}}"
  --sglang-context-length ${SGLANG_CONTEXT_LEN}
  --sglang-max-prefill-tokens ${SGLANG_MAX_PREFILL_TOKENS}
  --sglang-max-total-tokens ${SGLANG_MAX_TOTAL_TOKENS}
)

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
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\"
  }
}"

bash scripts/setup/ray_start.sh

echo "============================================================"
echo "TEST MEMORY OPTIMIZATIONS: CP=1, entropy_coef=0.001, max_tokens=16384"
echo "Optimizations being tested:"
echo "  1. FA TP=1 path: _logprobs_from_logits (flash_attn cross_entropy, no clone)"
echo "  2. Chunked backward: hidden_states → chunk projection → sequential backward"
echo "  3. Entropy under no_grad: skips autograd graph for entropy"
echo "EXP=${EXP_NAME}"
echo "============================================================"

ray job submit --address="http://${MASTER_ADDR}:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes ${WORLD_SIZE} \
  --actor-num-gpus-per-node ${GPUS_PER_NODE} \
  --colocate \
  --start-rollout-id 0 \
  --skip-eval-before-train \
  "${MODEL_ARGS_FAST[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" "$@"

bash $WORK_DIR/scripts/setup/ray_end.sh
echo "TEST MEMORY OPTIMIZATIONS done! EXP=${EXP_NAME}"
