#!/bin/bash
# MiniCPM5 16a3b short RL example, colocate.

:<<"EOF"
infra tuning ledger (MiniCPM5-MoE-16a3b @ 8k short, H100, 8-card default)
arch: 28 layers, GQA(KVheads=2, head_dim=128), MoE 160+1 topk=16; weight ~28GB/card (TP=1)
KV/token = 2(K,V) x 2 x 128 x 28 x 2B = 28 KiB   (1 GiB ~= 37k token)
changes (base -> tuned):
  rollout-num-gpus-per-engine    2 -> 1        (DP over TP: +38% throughput / -35% time at 8k)
  rollout_batch_size            64 -> 256      (32/card x 8 cards)
  max_running_requests         128 -> 192      (cheap KV supports high concurrency: +14% throughput)
  max_total_tokens          131072 -> 1536000  (mem0.88 allocates ~1.52M KV / 40.6GB)
  mem_fraction                0.75 -> 0.88
measured (per-card config): rollout_time 430s, 1672 tok/gpu/s (vs mr128: 544s/1470)
note: mem0.88 leaves ~8GB workspace; long-response batches may logprob-OOM -> lower mem/max_total
note: EP=8/CP=4 is training-side (8-card full RL), independent of per-engine inference unit
EOF

set -x
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CLUSTER=${CLUSTER:-ningxia_h100}

source scripts/setup/env.sh
bash scripts/setup/setup.sh
export SLIME_SCRIPT_NUM_GPUS=$GPUS_PER_NODE

BASE_DIR=${BASE_DIR:-/para_hdd/user/panyinxu/$PROJECT_NAME/$JOB_ID}
mkdir -p "$BASE_DIR"

export SGLANG_PATH=${WORK_DIR}/sglang
export ARG_MODE=true_on_policy
export WANDB_API_KEY=${WANDB_API_KEY:-i_dont_have_wandb_key}
USE_WANDB=${USE_WANDB:-0}

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

HF_MODEL_DIR=${HF_MODEL_DIR:-/user/panyinxu/models/job_362482_step_47000/}
mkdir -p /root/.cache/huggingface/modules/transformers_modules/
cp ${HF_MODEL_DIR}/configuration_minicpm.py \
   ${HF_MODEL_DIR}/modeling_minicpm.py \
   /root/.cache/huggingface/modules/transformers_modules/

export MODEL_PATH=${HF_MODEL_DIR}
export MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-/user/panyinxu/ckpts/minicpm5_362482_step47000}
export CKPT_PATH=${CKPT_PATH:-/data/checkpoints}
export PYTHONPATH=.:Megatron-LM/:mbridge:${SGLANG_PATH}/python:${WORK_DIR}/examples/retool_v2
source "${WORK_DIR}/scripts/minicpm5/16a3b/minicpm5_moe_16a3b_rope.sh"

MODEL_ARGS_SELECTED=()
skip_next=0
for a in "${MODEL_ARGS[@]}"; do
  if [ "$skip_next" -eq 1 ]; then
    skip_next=0
    continue
  fi
  if [ "$a" = "--moe-router-dtype" ] || [ "$a" = "--moe-token-dispatcher-type" ]; then
    skip_next=1
    continue
  fi
  MODEL_ARGS_SELECTED+=("$a")
done
MODEL_ARGS_SELECTED+=(--bf16 --moe-router-dtype fp32)

MODEL_TAG=minicpm5_moe_16a3b
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-4}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}   # DP over TP: ~35% faster at 8k; 28GB weight fits one card
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-8}
OPTIMIZER=${OPTIMIZER:-muon}
SGLANG_EXTRA_ARGS=(--sglang-disable-shared-experts-fusion --sglang-disable-overlap-schedule)


TASK_TAG=short_math_8k
# TRAIN_FILE=${TRAIN_FILE:-/user/yuzhuolei/rl_math/s10/rl_math_s10_mathverify_dpsk_filtered.jsonl}
TRAIN_FILE=${TRAIN_FILE:-/user/panyinxu/datasets/miles_data/eurus-rl/simple-math.jsonl}
TEST_FILE=${TEST_FILE:-"aime2025 /user/panyinxu/datasets/miles_data/aime-2025.jsonl aime2024 /user/panyinxu/datasets/miles_data/aime-2024.jsonl"}
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-256}   # 8 cards: bsz_per_card(32) x 8
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-2048}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-6144}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-8192}
SGLANG_CONTEXT_LEN=${SGLANG_CONTEXT_LEN:-8192}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
LOGPROBS_MAX_TOKENS_PER_GPU=${LOGPROBS_MAX_TOKENS_PER_GPU:-${MAX_TOKENS_PER_GPU}}
SGLANG_MAX_TOTAL_TOKENS=${SGLANG_MAX_TOTAL_TOKENS:-1536000}       # mem0.88 allocates ~1.52M KV (40.6GB)
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-192}   # per-engine; cheap KV(28KiB) supports high concurrency: +14% vs 128


export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${BASE_DIR}/exp_log/tensorboard/${MODEL_TAG}_${TASK_TAG}}

GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-2048}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.88}   # leaves ~8GB workspace; long-response batches risk logprob-OOM
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-${SGLANG_CONTEXT_LEN}}
SGLANG_SCHEDULE_CONSERVATIVENESS=${SGLANG_SCHEDULE_CONSERVATIVENESS:-1.2}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN}}
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${MODEL_TAG}_${TASK_TAG}_${RUN_TAG}"

CKPT_ARGS=(
  --hf-checkpoint $MODEL_PATH
  --load $MEGATRON_MODEL_PATH
  --save ${CKPT_PATH}/${EXP_NAME}/
  --save-interval ${SAVE_INTERVAL:-50}
  --finetune
  --no-load-optim
  --no-load-rng
)

ROLLOUT_ARGS=(
  --prompt-data $TRAIN_FILE
  --input-key ${INPUT_KEY:-prompt}
  --label-key ${LABEL_KEY:-label}
  --apply-chat-template
  --apply-chat-template-kwargs '{"enable_thinking": false}'
  --rollout-shuffle
  --num-rollout ${NUM_ROLLOUT}
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
  --rollout-max-prompt-len ${ROLLOUT_MAX_PROMPT_LEN}
  --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
  --rollout-max-context-len ${ROLLOUT_MAX_CONTEXT_LEN}
  --rollout-temperature ${ROLLOUT_TEMPERATURE:-1.0}
  --rollout-top-p ${ROLLOUT_TOP_P:-1.0}
  --global-batch-size ${GLOBAL_BATCH_SIZE}
  --balance-data
)

ROLLOUT_ARGS+=(
  --rm-type math
  --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)
CUSTOM_ARGS=()

PERF_ARGS=(
  --train-backend megatron
  --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE}
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE}
  --expert-tensor-parallel-size 1
  --attention-backend ${ATTENTION_BACKEND:-flash}
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
  --log-probs-max-tokens-per-gpu ${LOGPROBS_MAX_TOKENS_PER_GPU}
  --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --disable-grpo-std-normalization
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --calculate-per-token-loss
  --eps-clip 0.2
  --eps-clip-high 0.28
)

GRPO_ARGS+=(
  --use-rollout-routing-replay
  --use-miles-router
  --use-tis
  --custom-tis-function-path miles.backends.training_utils.loss.icepop_function
  --tis-clip-low 0.5
  --tis-clip 5.0
)


if [ "$OPTIMIZER" = "muon" ]; then
  OPTIMIZER_ARGS=(
    --optimizer muon
    --lr ${LR:-1e-6}
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --muon-matched-adamw-rms 0.2
    --muon-momentum 0.95
    --muon-ns-steps 5
    --muon-coefficient-type simple
  )
else
  OPTIMIZER_ARGS=(
    --optimizer adam
    --lr ${LR:-1e-6}
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
  )
fi

WANDB_ARGS=(--use-tensorboard)
if [ "${USE_WANDB}" = "1" ]; then
  WANDB_ARGS+=(--use-wandb --wandb-project ${WANDB_PROJECT:-miles-minicpm5})
  if [ -n "${WANDB_TEAM:-}" ]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
  fi
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION}
  --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}
  --sglang-schedule-conservativeness ${SGLANG_SCHEDULE_CONSERVATIVENESS}
  --sglang-attention-backend fa3
  --sglang-json-model-override-args "{\"max_position_embeddings\": ${SGLANG_CONTEXT_LEN}}"
  --sglang-context-length ${SGLANG_CONTEXT_LEN}
  --sglang-max-prefill-tokens ${SGLANG_MAX_PREFILL_TOKENS}
  --sglang-max-total-tokens ${SGLANG_MAX_TOTAL_TOKENS}
  "${SGLANG_EXTRA_ARGS[@]}"
)

EVAL_ARGS=(
  --eval-prompt-data $TEST_FILE
  --eval-interval ${EVAL_INTERVAL:-5}
  --eval-temperature ${EVAL_TEMPERATURE:-0.6}
  --eval-max-response-len ${EVAL_MAX_RESPONSE_LEN}
  --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL_PROMPT:-1}
)

mkdir -p ${CKPT_PATH}/${EXP_NAME}/dump_details
MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --moe-token-dispatcher-type alltoall
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\":  \".:./Megatron-LM:./mbridge:${SGLANG_PATH}/python:${WORK_DIR}/examples/retool_v2:/root/.cache/huggingface/modules\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\",
    \"NVTE_DEBUG\": \"${NVTE_DEBUG:-0}\",
    \"NVTE_DEBUG_LEVEL\": \"${NVTE_DEBUG_LEVEL:-0}\",
    \"NVTE_FUSED_ATTN\": \"0\",
    \"NVTE_UNFUSED_ATTN\": \"0\",
    \"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION\": \"10000\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"TMPDIR\": \"${TMPDIR}\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\"
  }
}"

bash scripts/setup/ray_start.sh

echo "============================================================"
echo "${MODEL_TAG} ${TASK_TAG}: colocate RL"
echo "rollouts=${NUM_ROLLOUT}, batch=${ROLLOUT_BATCH_SIZE}, samples=${N_SAMPLES_PER_PROMPT}"
echo "TP=${TENSOR_MODEL_PARALLEL_SIZE}, CP=${CONTEXT_PARALLEL_SIZE}, EP=${EXPERT_MODEL_PARALLEL_SIZE}, sglang_tp=${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "prompt_len=${ROLLOUT_MAX_PROMPT_LEN}, response_len=${ROLLOUT_MAX_RESPONSE_LEN}, context_len=${ROLLOUT_MAX_CONTEXT_LEN}"
echo "partial_rollout=0"
echo "EXP=${EXP_NAME}"
echo "BASE_DIR=${BASE_DIR}"
echo "============================================================"

ray job submit --address="http://${MASTER_ADDR}:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes ${WORLD_SIZE} \
  --actor-num-gpus-per-node ${GPUS_PER_NODE} \
  --colocate \
  --start-rollout-id 0 \
  --skip-eval-before-train \
  "${MODEL_ARGS_SELECTED[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  "${MISC_ARGS[@]}" "$@"

bash $WORK_DIR/scripts/setup/ray_end.sh
echo "${MODEL_TAG} ${TASK_TAG} done! EXP=${EXP_NAME}"
