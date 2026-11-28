#!/usr/bin/env bash
# Single-node (2xH100) GRPO run for the local-backend coding-agent RL stack.
#
# Arms are selected with env vars BEFORE launching:
#   TTL_SEC   sandbox instance TTL          (defective: < agent budget; control: aligned)
#   ARM_TAG   output subdirectory name
# Everything else is shared between arms.

set -euo pipefail

# ---------------- run identity ----------------
DURABLE="${DURABLE:-$AUTORLBENCH_WORKSPACE}"
WS_DIR="${WS_DIR:-$DURABLE/workspace}"
ASSETS="${ASSETS:-$DURABLE/assets}"
HF_CKPT="${HF_CKPT:-$ASSETS/Qwen3-4B-Instruct-2507}"

ARM_TAG="${ARM_TAG:-pilot}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$DURABLE/runs/${ARM_TAG}}"
mkdir -p "${RUN_ROOT}/rollout_dumps" "${RUN_ROOT}/episodes" "${RUN_ROOT}/sbx"
LOG_FILE="${RUN_ROOT}/run.log"

# ---------------- fault knobs ----------------
# Ops caps sandbox instance lifetime to control host cost. The agent-side time
# budget is a separate knob (SWE_AGENT_TIME_BUDGET_SEC below); see notes.md.
export SLIME_AGENT_SANDBOX_TTL_SEC="${TTL_SEC:-0}"
export SWE_AGENT_TIME_BUDGET_SEC="${AGENT_BUDGET_SEC:-600}"
export SWE_EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-180}"
export SWE_ROLLOUT_GUARD_SEC="$((SWE_AGENT_TIME_BUDGET_SEC + SWE_EVAL_TIMEOUT_SEC + 300))"

# ---------------- sandbox backend ----------------
# "local": per-episode process sandboxes on the trainer host.
# "e2b":   remote sandboxes (AGS-compatible gateway); connection env comes
#         from E2B_API_KEY / E2B_API_URL / E2B_DOMAIN, routing from
#         TENCENT_AGS_* below, and the agent callback goes through the
#         platform expose URL (ADAPTER_PUBLIC_URL + ADAPTER_PATH_PREFIX).
export SLIME_SANDBOX_BACKEND="${SLIME_SANDBOX_BACKEND:-local}"
export SLIME_LOCAL_SANDBOX_ROOT="${RUN_ROOT}/sbx"
export SLIME_INSTANCE_STORE="${WS_DIR}"
export SWE_EPISODE_ROOT="${RUN_ROOT}/episodes"
if [ "${SLIME_SANDBOX_BACKEND}" = "e2b" ]; then
    # TTL_SEC flows through SLIME_AGENT_SANDBOX_TTL_SEC for every backend
    # (set above); persistent AGS instances ignore server-side lifetimes, so
    # the cap is enforced by the backend's own recycle watchdog.
    export TENCENT_AGS_TEMPLATE="${TENCENT_AGS_TEMPLATE:-openclaw-new}"
    export TENCENT_AGS_IMAGE_REGISTRY_TYPE="${TENCENT_AGS_IMAGE_REGISTRY_TYPE:-enterprise}"
    export TENCENT_AGS_SANDBOX_CPU="${TENCENT_AGS_SANDBOX_CPU:-2}"
    export TENCENT_AGS_SANDBOX_MEM="${TENCENT_AGS_SANDBOX_MEM:-4Gi}"
    export TENCENT_AGS_REQUEST_TIMEOUT="${TENCENT_AGS_REQUEST_TIMEOUT:-300}"
    # the gateway's envd cannot exec non-root users through the /root/...
    # workspace; production rollouts ran as root
    export E2B_FORCE_USER="${E2B_FORCE_USER:-root}"
    export ADAPTER_PATH_PREFIX="${ADAPTER_PATH_PREFIX:-}"
    export ADAPTER_PUBLIC_URL="${ADAPTER_PUBLIC_URL:-}"
    : "${E2B_API_KEY:?SLIME_SANDBOX_BACKEND=e2b requires E2B_API_KEY}"
    : "${E2B_DOMAIN:?SLIME_SANDBOX_BACKEND=e2b requires E2B_DOMAIN}"
    export E2B_API_URL="${E2B_API_URL:-https://api.${E2B_DOMAIN}}"
fi

# ---------------- harness ----------------
export SWE_AGENT="${SWE_AGENT:-claude_code}"
export SLIME_AGENT_CC_TARBALL="${ASSETS}/anthropic-ai-claude-code-linux-x64-2.1.295.tgz"
export SLIME_AGENT_NODE_TARBALL="${SLIME_AGENT_CC_TARBALL}"  # unused on the pre-provisioned local backend
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-8}"
export SWE_BOOT_RETRIES=2

export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-127.0.0.1}"
export ADAPTER_BIND_HOST=0.0.0.0
# e2b: the port must match the platform expose that forwards to this pod;
# local: ephemeral is fine.
if [ "${SLIME_SANDBOX_BACKEND}" = "e2b" ]; then
    export ADAPTER_PORT="${ADAPTER_PORT:-18001}"
else
    export ADAPTER_PORT="${ADAPTER_PORT:-0}"
fi

SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":40000}'
export SLIME_AGENT_CC_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disable-slash-commands"
# the local backend runs everything as root; CC requires this to allow
# bypassPermissions under root (container/CI semantics)
export SLIME_AGENT_CC_EXTRA_ENVS='{"IS_SANDBOX": "1"}'

POD_IP="${POD_IP:-$(hostname -I | awk '{print $1}')}"
export no_proxy="127.0.0.1,localhost,${POD_IP}"
export NO_PROXY="${no_proxy}"

# ---------------- training hyperparams ----------------
NUM_GPUS="${NUM_GPUS:-2}"
NUM_ROLLOUT="${NUM_ROLLOUT:-80}"
ROLLOUT_BS="${ROLLOUT_BS:-4}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-49152}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
LR="${LR:-1e-6}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${WS_DIR}/slime/scripts/models/qwen3-4B-Instruct-2507.sh"

TORCH_DIST="${TORCH_DIST:-$ASSETS/Qwen3-4B-Instruct-2507_torch_dist}"
CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --save "/tmp/ckpt_${ARM_TAG}"
   --save-interval 10
)
if ls "/tmp/ckpt_${ARM_TAG}" 2>/dev/null | grep -q .; then
   # resume from this run's own training checkpoints
   CKPT_ARGS+=(--load "/tmp/ckpt_${ARM_TAG}")
else
   # fresh arm: start from the converted torch_dist base weights
   CKPT_ARGS+=(--load "${TORCH_DIST}")
fi

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data "${PROMPT_DATA:-${WS_DIR}/data/swe_train.jsonl}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BS}"
   --n-samples-per-prompt "${N_SAMPLES}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${MAX_RESPONSE_LEN}"
   --rollout-temperature 1.0
   --global-batch-size "$((ROLLOUT_BS * N_SAMPLES))"
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu $((MAX_CONTEXT_LEN / 2))
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.55
   --sglang-tool-call-parser qwen25
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

# ---------------- ray cluster ----------------
pkill -9 -f "sglang[.]launch" 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
sleep 2

ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port 8265 \
   --temp-dir "/tmp/ray_${ARM_TAG}"
sleep 5

export WS_DIR
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json, os
keys = (
    "SLIME_SANDBOX_BACKEND", "SLIME_LOCAL_SANDBOX_ROOT", "SLIME_INSTANCE_STORE",
    "SWE_EPISODE_ROOT",
    "SLIME_AGENT_SANDBOX_TTL_SEC", "SLIME_AGENT_SANDBOX_LIFETIME_SEC",
    "TENCENT_AGS_DOMAIN", "TENCENT_AGS_TEMPLATE", "TENCENT_AGS_IMAGE_REGISTRY_TYPE",
    "TENCENT_AGS_IMAGE_REWRITE", "TENCENT_AGS_SANDBOX_CPU", "TENCENT_AGS_SANDBOX_MEM",
    "TENCENT_AGS_REQUEST_TIMEOUT",
    "E2B_API_KEY", "E2B_API_URL", "E2B_DOMAIN", "E2B_PROXY", "E2B_FORCE_USER",
    "ADAPTER_PATH_PREFIX", "ADAPTER_PUBLIC_URL",
    "SWE_AGENT_TIME_BUDGET_SEC", "SWE_EVAL_TIMEOUT_SEC", "SWE_ROLLOUT_GUARD_SEC",
    "SWE_AGENT", "SLIME_AGENT_CC_TARBALL", "SLIME_AGENT_NODE_TARBALL",
    "SLIME_AGENT_CC_EXTRA_ARGS", "SLIME_AGENT_CC_EXTRA_ENVS",
    "SWE_BOOT_CONCURRENCY", "SWE_BOOT_RETRIES",
    "ADAPTER_PUBLIC_HOST", "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "no_proxy", "NO_PROXY", "WANDB_MODE", "PYTHONUNBUFFERED",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["MASTER_ADDR"] = "127.0.0.1"
env["MASTER_PORT"] = "6379"
env["PYTHONPATH"] = os.environ["WS_DIR"] + "/slime:/root/Megatron-LM"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
print(json.dumps({"env_vars": env}))
PY
)

echo "======================================================================"
echo "RUN_ROOT=${RUN_ROOT}"
echo "TTL_SEC=${SLIME_AGENT_SANDBOX_TTL_SEC}  AGENT_BUDGET=${SWE_AGENT_TIME_BUDGET_SEC}  EVAL_TIMEOUT=${SWE_EVAL_TIMEOUT_SEC}"
echo "log: ${LOG_FILE}"
echo "======================================================================"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u "${WS_DIR}/slime/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --rollout-num-gpus "${NUM_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
