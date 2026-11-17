#!/usr/bin/env bash
# Qwen3.5-4B OpenClaw GRPO 训练主入口, 支持 sync(colocate) 与 fully-async 两种
# TRAINING_MODE。以 contributor 的 run_aux_model 为模板 Qwen 化:
# 模型/解析器/路径换成reference, async/TIS/长度上限等机制原样保留。

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}"
MEGATRON_DIR="${MEGATRON_DIR:-/opt/Megatron-LM}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
RAY_BIN="${RAY_BIN:-$(dirname "${PYTHON_BIN}")/ray}"
E2B_ENV_FILE=""

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: missing Python: ${PYTHON_BIN}" >&2; exit 1; }
[[ -x "${RAY_BIN}" ]] || { echo "ERROR: missing Ray CLI: ${RAY_BIN}" >&2; exit 1; }

# Job-level environment variables take precedence over the shared fallback file.
declare -A JOB_ENV=()
for name in OPENCLAW_FORK_THRESHOLD_TOKENS OPENCLAW_E2B_SETUP_ATTEMPTS \
  OPENCLAW_E2B_SETUP_BACKOFF_BASE_SEC OPENCLAW_E2B_ACTOR_COUNT OPENCLAW_E2B_ACTOR_MAX_CONCURRENCY \
  SEARCH_PROVIDER SERP_API_KEY UDA_READER_ENDPOINT UDA_READER_API_KEY \
  ADAPTER_FORWARDER_ENABLED ADAPTER_TARGET_HOST ADAPTER_TARGET_PORT ADAPTER_PORT ADAPTER_FORWARDER_LOG_FILE \
  SLIME_ADAPTER_STAGE_LOG OPENCLAW_SAVE_TRANSCRIPT OPENCLAW_RUNTIME_DIAGNOSTICS_ENABLED \
  OPENCLAW_RUNTIME_DIAGNOSTICS_TIMEOUT_SEC OPENCLAW_REMOTE_CLEANUP_TIMEOUT_SEC \
  OPENCLAW_ADAPTER_SOCKET_HEALTH_SEC OPENCLAW_AGENT_LOG_TAIL_CHARS \
  OPENCLAW_ADVISOR OPENCLAW_ADVISOR_PLUGIN_DIR ADVISOR_STYLE \
  CLOUD_MODEL CLOUD_BASE_URL CLOUD_API_KEY OPENCLAW_ADVISOR_COST_MODEL \
  OPENCLAW_ADVISOR_REWARD_MODE OPENCLAW_ADVISOR_PENALTY_PER_CALL OPENCLAW_ADVISOR_PENALTY_CAP \
  OPENCLAW_ADVISOR_FORCE_CLOUD OPENCLAW_ADVISOR_COST_BUDGET_RATIO OPENCLAW_ADVISOR_COST_DUAL_LR \
  OPENCLAW_ADVISOR_COST_LAMBDA_INIT OPENCLAW_ADVISOR_COST_LAMBDA_MAX \
  OPENCLAW_ADVISOR_COST_NONCACHED_INPUT_PER_1K_RMB OPENCLAW_ADVISOR_COST_CACHED_INPUT_PER_1K_RMB \
  OPENCLAW_ADVISOR_COST_OUTPUT_PER_1K_RMB; do
  if [[ -v "${name}" ]]; then
    JOB_ENV["${name}"]="${!name}"
  fi
done

# SwanLab >= 0.9 reserves SWANLAB_PROJECT for a nested settings object.
export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-${SWANLAB_PROJECT:-slime-openclaw-rl}}"
unset SWANLAB_PROJECT
for name in "${!JOB_ENV[@]}"; do
  export "${name}=${JOB_ENV[${name}]}"
done

required_web_env=(
  SEARCH_PROVIDER SERP_API_KEY
  UDA_READER_ENDPOINT UDA_READER_API_KEY
)
missing_web_env=()
for name in "${required_web_env[@]}"; do
  [[ -n "${!name:-}" ]] || missing_web_env+=("${name}")
done
if ((${#missing_web_env[@]})); then
  echo "ERROR: OpenClaw web tools require these environment variables: ${missing_web_env[*]}" >&2
  echo "Set them in ${E2B_ENV_FILE} or export them before launching training." >&2
  exit 1
fi
[[ "${SEARCH_PROVIDER}" == "serp" ]] || {
  echo "ERROR: SEARCH_PROVIDER must be 'serp', got '${SEARCH_PROVIDER}'" >&2
  exit 1
}
export UDA_READER_WORK_PROJECT_ID="${UDA_READER_WORK_PROJECT_ID:-aux_model}"
export UDA_READER_PROXY_PROJECT="${UDA_READER_PROXY_PROJECT:-uda-reader}"
export UDA_READER_PROXY_SOURCES="${UDA_READER_PROXY_SOURCES:-51}"

# ============ paths ============
HF_CHECKPOINT="${HF_CHECKPOINT:-${SLIME_DIR}/models/Qwen3.5-4B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-${SLIME_DIR}/models/Qwen3.5-4B_torch_dist}"
CLAWGYM_DATA_DIR="${CLAWGYM_DATA_DIR:-${SLIME_DIR}/data/clawgym_train}"

RUN_NAME="${RUN_NAME:-qwen35_4b_openclawrl_async_128k_8actor_24rollout}"
RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/fanout}"
# 数据对齐 CPM 版: manifest 生成到 RUN_ROOT 下, 而不是脚本目录里的静态副本。
PROMPT_DATA="${PROMPT_DATA:-${RUN_ROOT}/data/clawgym_train.jsonl}"
TRAINING_MODE="${TRAINING_MODE:-async}"
mkdir -p "${RUN_ROOT}/rollout_dumps" "${RUN_ROOT}/tracking" "$(dirname "${PROMPT_DATA}")"

[[ -f "${HF_CHECKPOINT}/config.json" ]] || { echo "ERROR: invalid HF checkpoint: ${HF_CHECKPOINT}" >&2; exit 1; }
[[ -f "${REF_MODEL_PATH}/latest_checkpointed_iteration.txt" ]] || {
  echo "ERROR: invalid torch-dist reference checkpoint: ${REF_MODEL_PATH}" >&2
  exit 1
}

export PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
cd "${SLIME_DIR}"
"${PYTHON_BIN}" -m examples.openclaw_rl.prepare_manifest \
  --dataset-dir "${CLAWGYM_DATA_DIR}" \
  --output "${PROMPT_DATA}"

MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-4B.sh}"
[[ -f "${MODEL_CONFIG_SCRIPT}" ]] || {
  echo "ERROR: missing model configuration: ${MODEL_CONFIG_SCRIPT}" >&2
  exit 1
}
source "${MODEL_CONFIG_SCRIPT}"

# ============ regular GRPO batch ============
# 对齐 CPM async run: global batch 16x8=128, 温度 1.0, 500 步。
CONTEXT_WINDOW="${CONTEXT_WINDOW:-131072}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-32768}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
NUM_ROLLOUT="${NUM_ROLLOUT:-500}"

# ============ model parallelism ============
# 8 卡 actor: TP1*PP1*CP8=8, DP=1。用满 CP 换 token 预算 —— micro-batch 预算
# = 8192*8 = 65536, 是 PP2/CP4 那版(32768)的两倍, 超预算样本(会被 first_fit_pack
# 单独成 bin, 见 utils/dp_schedule.py 的不变量)因此少一半, OOM 面也小一半。
#
# 代价: PP=1 意味着 32 层的权重和 fp32 梯度都不再按 stage 切开, 每卡比 PP2 多扛
# 约 12G(权重 8G + 梯度 16G 的一半)。优化器已 CPU offload, 激活有 full recompute
# 兜着, 80G 卡上估算放得下, 但 4B 上没实测过 —— 头几步盯着显存, OOM 就退回
# PP_SIZE=2 CP_SIZE=4, 或上 ACTOR_NUM_NODES=2 ROLLOUT_NUM_GPUS=16 保持 CP8。
#
# CP 必须留在单节点内: CP=16 跨节点时 ring attention 的 p2p 在 backward 会死锁,
# 触发 600s NCCL watchdog(2026-08-01 记录)。
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-1}"
ETP_SIZE="${ETP_SIZE:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"

if [[ -n "${ACTOR_NUM_GPUS_PER_NODE:-}" ]]; then
  GPUS_PER_NODE="${GPUS_PER_NODE:-${ACTOR_NUM_GPUS_PER_NODE}}"
else
  GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | awk '/^GPU [0-9]+:/ {n++} END {print n + 0}')}"
fi
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${GPUS_PER_NODE}}"
ACTOR_TOTAL_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${ACTOR_TOTAL_GPUS}}"
MODEL_PARALLEL_SIZE=$((TP_SIZE * PP_SIZE * CP_SIZE))
ACTOR_DP_SIZE=$((ACTOR_TOTAL_GPUS / MODEL_PARALLEL_SIZE))

for name in CONTEXT_WINDOW ROLLOUT_MAX_RESPONSE_LEN ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT \
  NUM_ROLLOUT TP_SIZE PP_SIZE CP_SIZE MAX_TOKENS_PER_GPU ACTOR_NUM_NODES \
  ACTOR_NUM_GPUS_PER_NODE ROLLOUT_NUM_GPUS; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: ${name} must be a positive integer, got ${!name}" >&2; exit 1; }
done
(( ACTOR_TOTAL_GPUS % (TP_SIZE * PP_SIZE * CP_SIZE) == 0 )) || {
  echo "ERROR: total actor GPUs ${ACTOR_TOTAL_GPUS} must be divisible by TP*PP*CP=$((TP_SIZE * PP_SIZE * CP_SIZE))" >&2
  exit 1
}
(( GLOBAL_BATCH_SIZE % ACTOR_DP_SIZE == 0 )) || {
  echo "ERROR: global batch ${GLOBAL_BATCH_SIZE} must be divisible by actor DP ${ACTOR_DP_SIZE}" >&2
  exit 1
}
case "${TRAINING_MODE}" in
  sync)
    (( ROLLOUT_NUM_GPUS == ACTOR_TOTAL_GPUS )) || {
      echo "ERROR: sync colocated mode requires ROLLOUT_NUM_GPUS=${ACTOR_TOTAL_GPUS}" >&2
      exit 1
    }
    ;;
  async)
    CLUSTER_TOTAL_GPUS=$(( ${WORLD_SIZE:-0} * GPUS_PER_NODE ))
    (( CLUSTER_TOTAL_GPUS > 0 && ACTOR_TOTAL_GPUS + ROLLOUT_NUM_GPUS == CLUSTER_TOTAL_GPUS )) || {
      echo "ERROR: async mode requires actor (${ACTOR_TOTAL_GPUS}) + rollout (${ROLLOUT_NUM_GPUS}) = cluster (${CLUSTER_TOTAL_GPUS})" >&2
      exit 1
    }
    ;;
  *)
    echo "ERROR: TRAINING_MODE must be sync or async, got ${TRAINING_MODE}" >&2
    exit 1
    ;;
esac
(( ROLLOUT_MAX_RESPONSE_LEN < CONTEXT_WINDOW )) || {
  echo "ERROR: response length must be smaller than context length" >&2
  exit 1
}

# ============ OpenClaw / E2B ============
export E2B_TEMPLATE="${E2B_TEMPLATE:-openclaw-new}"
export E2B_IMAGE="${E2B_IMAGE:-service.example.invalid/code-interpreter/wildclawbench-ubuntu:v1.3}"
export E2B_IMAGE_REGISTRY_TYPE="${E2B_IMAGE_REGISTRY_TYPE:-enterprise}"
export SLIME_AGENT_SANDBOX_RPC_RETRIES="${SLIME_AGENT_SANDBOX_RPC_RETRIES:-8}"
# ---- the only two length limits ------------------------------------------
# 1) per request  : OPENCLAW_MAX_TOKENS caps max_new_tokens for one SGLang call.
# 2) whole session: OPENCLAW_CONTEXT_WINDOW is passed to open_session() as
#    max_context_tokens; the adapter truncates the trajectory there.
export OPENCLAW_CONTEXT_WINDOW="${CONTEXT_WINDOW}"
export OPENCLAW_MAX_TOKENS="${ROLLOUT_MAX_RESPONSE_LEN}"
# 撞满窗口被截断的 session 保留 grader 分数(CPM 实测撞窗样本均分反而更高)。
export OPENCLAW_ZERO_REWARD_ON_CONTEXT_OVERFLOW="${OPENCLAW_ZERO_REWARD_ON_CONTEXT_OVERFLOW:-0}"
export OPENCLAW_ENABLE_THINKING="${OPENCLAW_ENABLE_THINKING:-1}"
export OPENCLAW_REPLAY_DEBUG="${OPENCLAW_REPLAY_DEBUG:-0}"
export OPENCLAW_FORK_THRESHOLD_TOKENS="${OPENCLAW_FORK_THRESHOLD_TOKENS:-1024}"
# 1500s(25min)。原 1800s 时实测:~19% 的 episode 是被这堵墙砍掉的, 且它们的中位
# 轮数 112(正常 episode 只有 24), 每轮耗时却一样 —— 是在刷轮次, 不是单轮变慢。
# 一个 group 要等齐 8 条采样, 含一条钉子户的概率就到八成, step 时间基本被这堵墙定死。
export OPENCLAW_AGENT_TIMEOUT_SEC="${OPENCLAW_AGENT_TIMEOUT_SEC:-1500}"
export OPENCLAW_REWARD_TIMEOUT_SEC="${OPENCLAW_REWARD_TIMEOUT_SEC:-180}"
export OPENCLAW_E2B_SETUP_ATTEMPTS="${OPENCLAW_E2B_SETUP_ATTEMPTS:-3}"
export OPENCLAW_E2B_SETUP_BACKOFF_BASE_SEC="${OPENCLAW_E2B_SETUP_BACKOFF_BASE_SEC:-1}"
export OPENCLAW_E2B_ACTOR_MAX_CONCURRENCY="${OPENCLAW_E2B_ACTOR_MAX_CONCURRENCY:-8}"

export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-6379}"
HOST_JOB_ID="$(hostname | sed -nE 's/^devspace-job-.*-([0-9]+)-main-[0-9]+$/\1/p')"
MODEL_JOB_ID="${MODEL_JOB_ID:-${PAI_JOB_ID:-${JOB_ID:-${SLURM_JOB_ID:-${HOST_JOB_ID}}}}}"
if [[ -z "${ADAPTER_PUBLIC_URL:-}" && -n "${MODEL_JOB_ID}" ]]; then
  ADAPTER_PUBLIC_URL="https://service.example.invalid"
fi
export ADAPTER_PUBLIC_URL="${ADAPTER_PUBLIC_URL:-}"
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${MASTER_ADDR}}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-8666}"
if [[ -z "${ADAPTER_FORWARDER_ENABLED+x}" ]]; then
  if (( ACTOR_NUM_NODES > 1 )); then
    export ADAPTER_FORWARDER_ENABLED=1
  else
    export ADAPTER_FORWARDER_ENABLED=0
  fi
fi
export ADAPTER_TARGET_HOST="${ADAPTER_TARGET_HOST:-${MASTER_ADDR}}"
export ADAPTER_TARGET_PORT="${ADAPTER_TARGET_PORT:-${ADAPTER_PORT}}"
export ADAPTER_FORWARDER_LOG_FILE="${ADAPTER_FORWARDER_LOG_FILE:-${RUN_ROOT}/diagnostics/adapter_forwarder_rank${RANK:-0}_$(hostname).log}"
export SLIME_ADAPTER_STAGE_LOG="${SLIME_ADAPTER_STAGE_LOG:-0}"
export OPENCLAW_SAVE_TRANSCRIPT="${OPENCLAW_SAVE_TRANSCRIPT:-0}"
export OPENCLAW_ADAPTER_SOCKET_HEALTH_SEC="${OPENCLAW_ADAPTER_SOCKET_HEALTH_SEC:-0}"
export OPENCLAW_RUNTIME_DIAGNOSTICS_ENABLED="${OPENCLAW_RUNTIME_DIAGNOSTICS_ENABLED:-0}"
export OPENCLAW_RUNTIME_DIAGNOSTICS_TIMEOUT_SEC="${OPENCLAW_RUNTIME_DIAGNOSTICS_TIMEOUT_SEC:-30}"
export OPENCLAW_REMOTE_CLEANUP_TIMEOUT_SEC="${OPENCLAW_REMOTE_CLEANUP_TIMEOUT_SEC:-30}"
ADAPTER_FORWARDER_CTL="${ADAPTER_FORWARDER_CTL:-${SLIME_DIR}/examples/openclaw_rl/adapter_forwarder_ctl.sh}"

[[ "${ADAPTER_FORWARDER_ENABLED}" =~ ^(0|1)$ ]] || {
  echo "ERROR: ADAPTER_FORWARDER_ENABLED must be 0 or 1, got ${ADAPTER_FORWARDER_ENABLED}" >&2
  exit 1
}

[[ -n "${ADAPTER_PUBLIC_URL}" ]] || {
  echo "ERROR: cannot infer platform job id; set MODEL_JOB_ID or ADAPTER_PUBLIC_URL" >&2
  exit 1
}

# ============ checkpoint ============
CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_MODEL_PATH}"
  --save "${RUN_ROOT}/checkpoints"
  --save-interval "${SAVE_INTERVAL:-25}"
)
if [[ -n "${LOAD_CHECKPOINT:-}" ]]; then
  [[ -f "${LOAD_CHECKPOINT}/latest_checkpointed_iteration.txt" ]] || {
    echo "ERROR: not a resumable checkpoint: ${LOAD_CHECKPOINT}" >&2
    exit 1
  }
  CKPT_ARGS+=(--load "${LOAD_CHECKPOINT}")
elif [[ -f "${RUN_ROOT}/checkpoints/latest_checkpointed_iteration.txt" ]]; then
  echo "ERROR: ${RUN_ROOT} already contains a checkpoint." >&2
  echo "Set LOAD_CHECKPOINT=${RUN_ROOT}/checkpoints to resume, or choose a new RUN_ROOT." >&2
  exit 1
fi
SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-0}"
if [[ "${SAVE_RETAIN_INTERVAL}" != "0" ]]; then
  [[ "${SAVE_RETAIN_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: SAVE_RETAIN_INTERVAL must be 0 (keep all) or a positive integer" >&2
    exit 1
  }
  CKPT_ARGS+=(--save-retain-interval "${SAVE_RETAIN_INTERVAL}")
fi

# Reward post-process defaults to the preserved static reward. Advisor reward
# modes select their own implementation; both are supplied by the caller so
# this launcher stays mode-agnostic.
REWARD_POST_PROCESS_PATH="${REWARD_POST_PROCESS_PATH:-examples.openclaw_rl.generate.normalize_trajectory_rewards}"
CUSTOM_ADVANTAGE_ARGS=()
if [[ -n "${CUSTOM_ADVANTAGE_FUNCTION_PATH:-}" ]]; then
  CUSTOM_ADVANTAGE_ARGS+=(--custom-advantage-function-path "${CUSTOM_ADVANTAGE_FUNCTION_PATH}")
fi

ROLLOUT_ARGS=(
  --custom-generate-function-path examples.openclaw_rl.generate.generate
  --custom-reward-post-process-path "${REWARD_POST_PROCESS_PATH}"
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-context-len "${CONTEXT_WINDOW}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --rollout-top-p "${ROLLOUT_TOP_P:-1.0}"
  --rollout-top-k -1
  --num-steps-per-rollout 1
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --micro-batch-size 1
  --balance-data
  --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)
if [[ -n "${OVER_SAMPLING_BATCH_SIZE:-}" ]]; then
  ROLLOUT_ARGS+=(--over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}")
fi
if [[ -n "${DYNAMIC_SAMPLING_FILTER_PATH:-}" ]]; then
  ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi
if [[ "${TRAINING_MODE}" == "async" ]]; then
  ROLLOUT_ARGS+=(
    --rollout-function-path
    slime.rollout.fully_async_rollout.generate_rollout_fully_async
  )
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size "${EP_SIZE}"
  --expert-tensor-parallel-size "${ETP_SIZE}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-2048}"
)

if [[ "${RECOMPUTE_LOSS_FUNCTION:-0}" == "1" ]]; then
  PERF_ARGS+=(--recompute-loss-function)
fi

# Loss normalization. Default (flag absent) is per-rollout-mean. Set
# CALCULATE_PER_TOKEN_LOSS=1 for global token-mean (the CPM async recipe).
if [[ "${CALCULATE_PER_TOKEN_LOSS:-0}" == "1" ]]; then
  PERF_ARGS+=(--calculate-per-token-loss)
fi

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-coef 0.0
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
)
if [[ "${DISABLE_GRPO_STD_NORMALIZATION:-0}" == "1" ]]; then
  GRPO_ARGS+=(--disable-grpo-std-normalization)
fi

# Fully-async rollout lags behind the actor; apply token-level importance
# sampling and reject ratios outside the IcePop interval.
USE_TIS="${USE_TIS:-0}"
case "${USE_TIS}" in
  0) ;;
  1)
    CUSTOM_TIS_FUNCTION_PATH="${CUSTOM_TIS_FUNCTION_PATH:-slime.backends.megatron_utils.loss.icepop_function}"
    TIS_CLIP_LOW="${TIS_CLIP_LOW:-0.5}"
    TIS_CLIP="${TIS_CLIP:-5.0}"
    GRPO_ARGS+=(
      --use-tis
      --custom-tis-function-path "${CUSTOM_TIS_FUNCTION_PATH}"
      --tis-clip-low "${TIS_CLIP_LOW}"
      --tis-clip "${TIS_CLIP}"
    )
    ;;
  *)
    echo "ERROR: USE_TIS must be 0 or 1, got ${USE_TIS}" >&2
    exit 1
    ;;
esac

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR:-1e-6}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)
if [[ "${OPTIMIZER_CPU_OFFLOAD:-1}" == "1" ]]; then
  OPTIMIZER_ARGS+=(
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
  )
fi

SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-$(((GLOBAL_BATCH_SIZE + ROLLOUT_NUM_GPUS - 1) / ROLLOUT_NUM_GPUS))}"
SGLANG_ARGS=(
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  --rollout-num-gpus-per-engine 1
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.75}"
  --sglang-context-length "${CONTEXT_WINDOW}"
  --sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS:-${CONTEXT_WINDOW}}"
  --sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS:-524288}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-32}"
  --sglang-schedule-conservativeness "${SGLANG_SCHEDULE_CONSERVATIVENESS:-1.2}"
  --sglang-attention-backend fa3
  --sglang-tool-call-parser qwen3_coder
  --sglang-reasoning-parser qwen3
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
if [[ "${TRAINING_MODE}" == "sync" ]]; then
  MISC_ARGS+=(--colocate)
fi
if [[ "${DEBUG_ROLLOUT_ONLY:-0}" == "1" ]]; then
  MISC_ARGS+=(--debug-rollout-only)
fi

export SLIME_USE_SWANLAB="${SLIME_USE_SWANLAB:-1}"
export SWANLAB_MODE="${SWANLAB_MODE:-online}"
export WANDB_MODE="${WANDB_MODE:-offline}"
TRACKING_BACKEND="${TRACKING_BACKEND:-swanlab}"
case "${TRACKING_BACKEND}" in
  swanlab)
    TRACKING_ARGS=(
      --use-swanlab
      --swanlab-project "${SWANLAB_PROJ_NAME:-slime-openclaw-rl}"
      --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME:-${RUN_NAME}}"
      --swanlab-mode "${SWANLAB_MODE}"
      --swanlab-log-dir "${SWANLAB_LOG_DIR:-${RUN_ROOT}/tracking}"
    )
    [[ -n "${SWANLAB_WORKSPACE:-}" ]] && TRACKING_ARGS+=(--swanlab-workspace "${SWANLAB_WORKSPACE}")
    [[ -n "${SWANLAB_GROUP:-}" ]] && TRACKING_ARGS+=(--swanlab-group "${SWANLAB_GROUP}")
    [[ -n "${SWANLAB_RUN_ID:-}" ]] && TRACKING_ARGS+=(--swanlab-run-id "${SWANLAB_RUN_ID}")
    ;;
  wandb)
    TRACKING_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT:-slime-openclaw-rl}"
      --wandb-group "${WANDB_GROUP:-${RUN_NAME}}"
      --wandb-dir "${RUN_ROOT}/tracking"
    )
    ;;
  *)
    echo "ERROR: TRACKING_BACKEND must be swanlab or wandb, got ${TRACKING_BACKEND}" >&2
    exit 1
    ;;
esac

export CUDA_DEVICE_MAX_CONNECTIONS=1
# sync(colocate) 模式禁止 expandable_segments(torch_memory_saver 硬性不兼容);
# async 的 actor 不与 SGLang 共卡, launcher 可以另行开启。
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export no_proxy="127.0.0.1,localhost,0.0.0.0,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST}${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"

echo "[openclaw-grpo] run=${RUN_NAME} root=${RUN_ROOT}"
echo "[openclaw-grpo] context=${CONTEXT_WINDOW} max_response=${ROLLOUT_MAX_RESPONSE_LEN} TP=${TP_SIZE} PP=${PP_SIZE} CP=${CP_SIZE} max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"
echo "[openclaw-grpo] length limits: per_request=${OPENCLAW_MAX_TOKENS} session=${OPENCLAW_CONTEXT_WINDOW}"
echo "[openclaw-grpo] zero_reward_on_context_overflow=${OPENCLAW_ZERO_REWARD_ON_CONTEXT_OVERFLOW} (0=keep grader score) load=${LOAD_CHECKPOINT:-<none>}"
echo "[openclaw-grpo] prompts/batch=${ROLLOUT_BATCH_SIZE} samples/prompt=${N_SAMPLES_PER_PROMPT} global_batch=${GLOBAL_BATCH_SIZE}"
echo "[openclaw-grpo] oversampling=${OVER_SAMPLING_BATCH_SIZE:-<none>} dynamic_filter=${DYNAMIC_SAMPLING_FILTER_PATH:-<none>} temperature=${ROLLOUT_TEMPERATURE:-1.0}"
echo "[openclaw-grpo] steps=${NUM_ROLLOUT} accepted_executions=$((NUM_ROLLOUT * GLOBAL_BATCH_SIZE)) thinking=${OPENCLAW_ENABLE_THINKING}"
echo "[openclaw-grpo] mode=${TRAINING_MODE} actor=${ACTOR_NUM_NODES}x${ACTOR_NUM_GPUS_PER_NODE} actor_gpus=${ACTOR_TOTAL_GPUS} rollout_gpus=${ROLLOUT_NUM_GPUS}"
echo "[openclaw-grpo] tool_parser=qwen3_coder reasoning_parser=qwen3"
echo "[openclaw-grpo] sglang_mem=${SGLANG_MEM_FRACTION:-0.75} max_total_tokens=${SGLANG_MAX_TOTAL_TOKENS:-524288} max_running_requests=${SGLANG_MAX_RUNNING_REQUESTS:-32}"
echo "[openclaw-grpo] replay_debug=${OPENCLAW_REPLAY_DEBUG} fork_threshold=${OPENCLAW_FORK_THRESHOLD_TOKENS} rollout_only=${DEBUG_ROLLOUT_ONLY:-0}"
echo "[openclaw-grpo] checkpoint_every=${SAVE_INTERVAL:-25} retain=${SAVE_RETAIN_INTERVAL:-0} (0=keep-all)"
echo "[openclaw-grpo] advisor=${OPENCLAW_ADVISOR:-0} reward_mode=${OPENCLAW_ADVISOR_REWARD_MODE:-static_penalty} cloud_model=${CLOUD_MODEL:-<unset>}"
echo "[openclaw-grpo] reward_post_process=${REWARD_POST_PROCESS_PATH}"
echo "[openclaw-grpo] custom_advantage=${CUSTOM_ADVANTAGE_FUNCTION_PATH:-<none>}"
echo "[openclaw-grpo] tis=${USE_TIS} function=${CUSTOM_TIS_FUNCTION_PATH:-<none>} clip=[${TIS_CLIP_LOW:-<none>},${TIS_CLIP:-<none>}]"
echo "[openclaw-grpo] grpo_std_normalization=$((1 - ${DISABLE_GRPO_STD_NORMALIZATION:-0})) per_token_loss=${CALCULATE_PER_TOKEN_LOSS:-0} optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD:-1} logprob_chunk=${LOG_PROBS_CHUNK_SIZE:-2048}"
echo "[openclaw-grpo] adapter_forwarder=${ADAPTER_FORWARDER_ENABLED} target=${ADAPTER_TARGET_HOST}:${ADAPTER_TARGET_PORT}"
echo "[openclaw-grpo] runtime_logging transcript=${OPENCLAW_SAVE_TRANSCRIPT} stage=${SLIME_ADAPTER_STAGE_LOG} socket_health=${OPENCLAW_ADAPTER_SOCKET_HEALTH_SEC} diagnostics=${OPENCLAW_RUNTIME_DIAGNOSTICS_ENABLED}"
echo "[openclaw-grpo] tracking=${TRACKING_BACKEND} project=${SWANLAB_PROJ_NAME:-slime-openclaw-rl} workspace=${SWANLAB_WORKSPACE:-<none>}"
if [[ "${TRAINING_MODE}" == "async" ]]; then
  TRAIN_ENTRYPOINT="${SLIME_DIR}/train_async.py"
  echo "[openclaw-grpo] async_driver=${TRAIN_ENTRYPOINT} rollout_function=slime.rollout.fully_async_rollout.generate_rollout_fully_async"
else
  TRAIN_ENTRYPOINT="${SLIME_DIR}/train.py"
fi

if [[ "${CONFIG_DRY_RUN:-0}" == "1" ]]; then
  echo "[openclaw-grpo] CONFIG_DRY_RUN=1, configuration is valid"
  exit 0
fi

# SKIP_RAY_LAUNCH=1 时(launch_128k_8gpu.sh 的 RANK-based 编排), ray 集群已在外部
# 起好, 这里跳过自带的 ray 启动, 直接提交 job 到现成 head。
if [[ "${SKIP_RAY_LAUNCH:-0}" != "1" ]]; then
  "${RAY_BIN}" stop --force || true
  "${RAY_BIN}" start --head --node-ip-address "${MASTER_ADDR}" --port "${MASTER_PORT}" \
    --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_JSON="$(${PYTHON_BIN} - <<'PY'
import json
import os

keys = (
    "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "PYTORCH_CUDA_ALLOC_CONF",
    "NVTE_BATCH_MHA_P2P_COMM",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
    "ADAPTER_PUBLIC_URL", "ADAPTER_PUBLIC_HOST", "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "ADAPTER_FORWARDER_ENABLED", "ADAPTER_TARGET_HOST", "ADAPTER_TARGET_PORT", "ADAPTER_FORWARDER_LOG_FILE",
    "E2B_API_URL", "E2B_DOMAIN", "E2B_VALIDATE_API_KEY",
    "E2B_TEMPLATE", "E2B_IMAGE", "E2B_IMAGE_REGISTRY_TYPE", "E2B_CPU", "E2B_MEMORY",
    "SEARCH_PROVIDER", "SEARCH_PLUGIN_DIR",
    "UDA_READER_PLUGIN_DIR", "UDA_READER_ENDPOINT", "UDA_READER_WORK_PROJECT_ID", "UDA_READER_PROXY_PROJECT", "UDA_READER_PROXY_SOURCES",
    "SLIME_AGENT_SANDBOX_RPC_RETRIES", "OPENCLAW_CONTEXT_WINDOW", "OPENCLAW_MAX_TOKENS",
    "OPENCLAW_ZERO_REWARD_ON_CONTEXT_OVERFLOW",
    "OPENCLAW_ENABLE_THINKING", "OPENCLAW_REPLAY_DEBUG", "OPENCLAW_FORK_THRESHOLD_TOKENS", "OPENCLAW_AGENT_TIMEOUT_SEC",
    "OPENCLAW_REWARD_TIMEOUT_SEC", "OPENCLAW_E2B_SETUP_ATTEMPTS",
    "OPENCLAW_E2B_SETUP_BACKOFF_BASE_SEC", "OPENCLAW_E2B_ACTOR_COUNT",
    "OPENCLAW_E2B_ACTOR_MAX_CONCURRENCY", "OPENCLAW_AGENT_LOG_TAIL_CHARS",
    "OPENCLAW_SAVE_TRANSCRIPT", "OPENCLAW_RUNTIME_DIAGNOSTICS_ENABLED",
    "OPENCLAW_RUNTIME_DIAGNOSTICS_TIMEOUT_SEC", "OPENCLAW_REMOTE_CLEANUP_TIMEOUT_SEC",
    "OPENCLAW_ADAPTER_SOCKET_HEALTH_SEC", "SLIME_ADAPTER_STAGE_LOG",
    "http_proxy", "https_proxy", "no_proxy", "NO_PROXY",
    "HTTP_PROXY", "HTTPS_PROXY",
    "OPENCLAW_ADVISOR", "OPENCLAW_ADVISOR_PLUGIN_DIR", "ADVISOR_STYLE",
    "CLOUD_MODEL", "CLOUD_BASE_URL", "OPENCLAW_ADVISOR_COST_MODEL",
    "OPENCLAW_ADVISOR_REWARD_MODE", "OPENCLAW_ADVISOR_PENALTY_PER_CALL", "OPENCLAW_ADVISOR_PENALTY_CAP",
    "OPENCLAW_ADVISOR_FORCE_CLOUD", "OPENCLAW_ADVISOR_COST_BUDGET_RATIO", "OPENCLAW_ADVISOR_COST_DUAL_LR",
    "OPENCLAW_ADVISOR_COST_LAMBDA_INIT", "OPENCLAW_ADVISOR_COST_LAMBDA_MAX",
    "OPENCLAW_ADVISOR_COST_NONCACHED_INPUT_PER_1K_RMB", "OPENCLAW_ADVISOR_COST_CACHED_INPUT_PER_1K_RMB",
    "OPENCLAW_ADVISOR_COST_OUTPUT_PER_1K_RMB",
    "SLIME_USE_SWANLAB", "SWANLAB_API_HOST", "SWANLAB_MODE", "WANDB_MODE",
    "SWANLAB_PROJ_NAME", "SWANLAB_EXPERIMENT_NAME", "SWANLAB_WORKSPACE", "SWANLAB_GROUP", "SWANLAB_RUN_ID",
)
print(json.dumps({"env_vars": {key: os.environ[key] for key in keys if key in os.environ}}))
PY
)"

RAY_JOB_WAIT_ARGS=()
if [[ "${RAY_JOB_NO_WAIT:-0}" == "1" ]]; then
  RAY_JOB_WAIT_ARGS+=(--no-wait)
fi

"${RAY_BIN}" job submit --address="http://127.0.0.1:8265" \
  "${RAY_JOB_WAIT_ARGS[@]}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON_BIN}" -u "${TRAIN_ENTRYPOINT}" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${CUSTOM_ADVANTAGE_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${TRACKING_ARGS[@]}" \
  2>&1 | tee -a "${RUN_ROOT}/train.log"
