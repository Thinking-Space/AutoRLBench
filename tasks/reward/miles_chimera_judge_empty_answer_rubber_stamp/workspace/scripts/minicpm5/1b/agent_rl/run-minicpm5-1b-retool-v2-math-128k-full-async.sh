#!/bin/bash
# MiniCPM5 1b agent RL example, full async.

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
FULLY_ASYNC_DIR="${WORK_DIR}/examples/fully_async"
export ARG_MODE=true_on_policy
export WANDB_API_KEY=${WANDB_API_KEY:-i_dont_have_wandb_key}
USE_WANDB=${USE_WANDB:-0}

if [ "${SKIP_KILL:-0}" != "1" ]; then
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -f "python3 train_async.py" || true
  pkill -f "python3 train.py" || true
  sleep 3
  pkill -9 ray || true
  pkill -f "python3 train_async.py" || true
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

HF_MODEL_DIR=${HF_MODEL_DIR:-/user/panyinxu/models/MiniCPM5-1B-SFT}
HF_MODULES_DIR=${HF_MODULES_DIR:-/root/.cache/huggingface/modules}
HF_DYNAMIC_MODULE_NAME="$(basename "${HF_MODEL_DIR}" | sed 's/\./_dot_/g; s/-/_/g')"
HF_DYNAMIC_MODULE_DIR="${HF_MODULES_DIR}/transformers_modules/${HF_DYNAMIC_MODULE_NAME}"
mkdir -p "${HF_DYNAMIC_MODULE_DIR}"
touch "${HF_MODULES_DIR}/transformers_modules/__init__.py" "${HF_DYNAMIC_MODULE_DIR}/__init__.py"
for py_file in configuration_minicpm.py modeling_minicpm.py; do
  if [ -f "${HF_MODEL_DIR}/${py_file}" ]; then
    cp "${HF_MODEL_DIR}/${py_file}" "${HF_DYNAMIC_MODULE_DIR}/"
  fi
done

export MODEL_PATH=${HF_MODEL_DIR}
export MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-/user/panyinxu/ckpts/MiniCPM5-1B-SFT_torch_dist}
export CKPT_PATH=${CKPT_PATH:-${BASE_DIR}/checkpoints}
export PYTHONPATH=.:Megatron-LM/:mbridge:${SGLANG_PATH}/python:${HF_MODULES_DIR}:${WORK_DIR}/examples/retool_v2:${FULLY_ASYNC_DIR}
source "${WORK_DIR}/scripts/minicpm5/1b/minicpm5-1b.sh"

MODEL_ARGS_SELECTED=("${MODEL_ARGS[@]}")
MODEL_TAG=minicpm5_1b
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-4}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
OPTIMIZER=${OPTIMIZER:-adam}
SGLANG_EXTRA_ARGS=()

MODEL_PARALLEL_GPUS=$((TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
CLUSTER_TOTAL_GPUS=$((WORLD_SIZE * GPUS_PER_NODE))
TRAIN_NODE_RATIO=${TRAIN_NODE_RATIO:-1}
ROLLOUT_NODE_RATIO=${ROLLOUT_NODE_RATIO:-3}
DEFAULT_ACTOR_NUM_NODES=$((WORLD_SIZE * TRAIN_NODE_RATIO / (TRAIN_NODE_RATIO + ROLLOUT_NODE_RATIO)))
if [ "${DEFAULT_ACTOR_NUM_NODES}" -lt 1 ]; then
  DEFAULT_ACTOR_NUM_NODES=1
fi
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-${DEFAULT_ACTOR_NUM_NODES}}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-${GPUS_PER_NODE}}
ACTOR_TOTAL_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
ROLLOUT_NUM_GPUS_TOTAL=${ROLLOUT_NUM_GPUS:-$((CLUSTER_TOTAL_GPUS - ACTOR_TOTAL_GPUS))}
EXPECTED_TOTAL_GPUS=$((ACTOR_TOTAL_GPUS + ROLLOUT_NUM_GPUS_TOTAL))
if [ "${ACTOR_NUM_GPUS_PER_NODE}" -lt "${MODEL_PARALLEL_GPUS}" ]; then
  echo "ERROR: ACTOR_NUM_GPUS_PER_NODE(${ACTOR_NUM_GPUS_PER_NODE}) must be >= tensor_model_parallel_size*context_parallel_size(${MODEL_PARALLEL_GPUS})"
  exit 1
fi
if [ "${ACTOR_NUM_GPUS_PER_NODE}" -gt "${GPUS_PER_NODE}" ]; then
  echo "ERROR: ACTOR_NUM_GPUS_PER_NODE(${ACTOR_NUM_GPUS_PER_NODE}) must be <= GPUS_PER_NODE(${GPUS_PER_NODE})"
  exit 1
fi
if [ "${ROLLOUT_NUM_GPUS_TOTAL}" -lt "${GPUS_PER_NODE}" ]; then
  echo "ERROR: ROLLOUT_NUM_GPUS(${ROLLOUT_NUM_GPUS_TOTAL}) must reserve at least one rollout node. Increase WORLD_SIZE or lower ACTOR_* settings."
  exit 1
fi
if [ "${EXPECTED_TOTAL_GPUS}" -ne "${CLUSTER_TOTAL_GPUS}" ]; then
  echo "ERROR: ACTOR_TOTAL_GPUS(${ACTOR_TOTAL_GPUS}) + ROLLOUT_NUM_GPUS(${ROLLOUT_NUM_GPUS_TOTAL}) != WORLD_SIZE*GPUS_PER_NODE(${CLUSTER_TOTAL_GPUS})"
  exit 1
fi

TASK_TAG=retool_v2_math_128k
TRAIN_FILE=${TRAIN_FILE:-/user/panyinxu/datasets/miles_data/dapo-math-17k-with-retool-v2-tools.jsonl}
TEST_FILE=${TEST_FILE:-"aime2025 /user/panyinxu/datasets/miles_data/aime-2025.jsonl aime2024 /user/panyinxu/datasets/miles_data/aime-2024.jsonl"}
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * 4))}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-2048}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-120000}
# Leave one token headroom: SGLang rejects input + max_new_tokens == context length.
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-130000}
SGLANG_CONTEXT_LEN=${SGLANG_CONTEXT_LEN:-131072}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16000}
LOGPROBS_MAX_TOKENS_PER_GPU=${LOGPROBS_MAX_TOKENS_PER_GPU:-$((MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE))}
SGLANG_MAX_TOTAL_TOKENS=${SGLANG_MAX_TOTAL_TOKENS:-524288}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-32}
GENERATE_MAX_TURNS=${GENERATE_MAX_TURNS:-16}

export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${BASE_DIR}/exp_log/tensorboard/${MODEL_TAG}_${TASK_TAG}}

GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-2048}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.75}
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
)

ROLLOUT_ARGS=(
  --rollout-function-path fully_async_rollout.generate_rollout_fully_async
  --prompt-data $TRAIN_FILE
  --input-key ${INPUT_KEY:-prompt}
  --label-key ${LABEL_KEY:-label}
  --tool-key ${TOOL_KEY:-tools}
  --apply-chat-template
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
  --custom-rm-path examples.retool_v2.tool_sandbox.reward_func
  --reward-key score
  --over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE}
  --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)
CUSTOM_ARGS=(
  --custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate
  --generate-tool-specs-path examples.retool_v2.tool_sandbox.tool_specs
  --generate-execute-tool-function-path examples.retool_v2.tool_sandbox.execute_tool
  --generate-tool-call-parser ${GENERATE_TOOL_CALL_PARSER:-minicpm4_xml}
  --generate-max-turns ${GENERATE_MAX_TURNS}
  --log-multi-turn
)

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
  --num-gpus-per-node ${GPUS_PER_NODE}
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
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\":  \".:./Megatron-LM:./mbridge:${SGLANG_PATH}/python:${WORK_DIR}/examples/retool_v2:${FULLY_ASYNC_DIR}:/root/.cache/huggingface/modules\",
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
echo "${MODEL_TAG} ${TASK_TAG}: full async RL"
echo "rollouts=${NUM_ROLLOUT}, batch=${ROLLOUT_BATCH_SIZE}, samples=${N_SAMPLES_PER_PROMPT}"
echo "TP=${TENSOR_MODEL_PARALLEL_SIZE}, CP=${CONTEXT_PARALLEL_SIZE}, EP=${EXPERT_MODEL_PARALLEL_SIZE}, sglang_tp=${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "train:rollout node ratio=${TRAIN_NODE_RATIO}:${ROLLOUT_NODE_RATIO}, actor_nodes=${ACTOR_NUM_NODES}"
echo "actor_gpus=${ACTOR_TOTAL_GPUS}, rollout_gpus=${ROLLOUT_NUM_GPUS_TOTAL}"
echo "prompt_len=${ROLLOUT_MAX_PROMPT_LEN}, response_len=${ROLLOUT_MAX_RESPONSE_LEN}, context_len=${ROLLOUT_MAX_CONTEXT_LEN}"
echo "partial_rollout=0"
echo "EXP=${EXP_NAME}"
echo "BASE_DIR=${BASE_DIR}"
echo "============================================================"

ray job submit --address="http://${MASTER_ADDR}:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train_async.py \
  --actor-num-nodes ${ACTOR_NUM_NODES} \
  --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE} \
  --rollout-num-gpus ${ROLLOUT_NUM_GPUS_TOTAL} \
  --start-rollout-id 0 \
  "${MODEL_ARGS_SELECTED[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  "${MISC_ARGS[@]}" "$@"

bash $WORK_DIR/scripts/setup/ray_end.sh
echo "${MODEL_TAG} ${TASK_TAG} done! EXP=${EXP_NAME}"
