#!/usr/bin/env bash
# Run one normal 8-GPU Miles/Harbor GRPO update and summarize the observation.
set -euo pipefail

WS="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${AUTORLBENCH_PREPARED:=$WS/.prepared}"
: "${AUTORLBENCH_RESULTS:=/tmp/miles-miniswe-harbor}"
mkdir -p "$AUTORLBENCH_RESULTS/experiment_logs"

if [ -f "$AUTORLBENCH_PREPARED/env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$AUTORLBENCH_PREPARED/env"
  set +a
fi

: "${MODEL_PATH:?set MODEL_PATH to the Qwen3-30B-A3B HF checkpoint}"
: "${SESSION_SERVER_EXTERNAL_HOST:?set SESSION_SERVER_EXTERNAL_HOST to a host the sandboxes can reach}"
: "${HARBOR_ENV_TYPE:=e2b}"
: "${SANDBOX_BACKEND:=tencent_ags}"
: "${MILES_SOURCE_DIR:?MILES_SOURCE_DIR is written by environment/prepare.sh}"
: "${MEGATRON_PATH:?MEGATRON_PATH is written by environment/prepare.sh}"
: "${MILES_MODEL_BASE:=$AUTORLBENCH_PREPARED/models}"
: "${MODEL_NAME:=Qwen3-30B-A3B-autorlbench}"
: "${MILES_PROMPT_DATA:=$WS/data/miles_prompts.jsonl}"
: "${HARBOR_TASKS_DIR:=$WS/harbor_tasks}"
: "${HARBOR_TRIALS_DIR:=$AUTORLBENCH_RESULTS/harbor_trials}"
: "${MILES_DUMP_DIR:=$AUTORLBENCH_RESULTS/dump_details}"
: "${MILES_HOST_IP:=0.0.0.0}"
: "${EXPECTED_LOGICAL_ROLLOUTS:=64}"
: "${RUN_MODE:=normal}"

if [ "$HARBOR_ENV_TYPE" != "e2b" ]; then
  echo "[run] this package currently expects HARBOR_ENV_TYPE=e2b; got $HARBOR_ENV_TYPE" >&2
  exit 2
fi

if [ "$SANDBOX_BACKEND" != "tencent_ags" ] && [ "$SANDBOX_BACKEND" != "e2b" ]; then
  echo "[run] SANDBOX_BACKEND must be tencent_ags or e2b; got $SANDBOX_BACKEND" >&2
  exit 2
fi

if [ "$SANDBOX_BACKEND" = "tencent_ags" ]; then
  : "${TENCENT_AGS_TEMPLATE:?set TENCENT_AGS_TEMPLATE to the fixed Tencent AGS template}"
  if [ -n "${TENCENT_AGS_DOMAIN:-}" ]; then
    E2B_DOMAIN="${E2B_DOMAIN:-$TENCENT_AGS_DOMAIN}"
    export E2B_DOMAIN
  fi
  if [ -n "${TENCENT_AGS_URL:-}" ]; then
    E2B_API_URL="${E2B_API_URL:-$TENCENT_AGS_URL}"
    export E2B_API_URL
  fi
  if [ -z "${E2B_API_URL:-}" ] && [ -n "${E2B_DOMAIN:-}" ]; then
    E2B_API_URL="https://api.${E2B_DOMAIN}"
    export E2B_API_URL
  fi
  if [ -z "${E2B_DOMAIN:-}" ] && [ -z "${E2B_SANDBOX_URL:-}" ]; then
    echo "[run] Tencent AGS needs TENCENT_AGS_DOMAIN (or E2B_SANDBOX_URL for a custom data plane)" >&2
    exit 2
  fi
  : "${TENCENT_AGS_IMAGE_REGISTRY_TYPE:=enterprise}"
  : "${TENCENT_AGS_FORCE_HTTP1:=0}"
  : "${TENCENT_AGS_AGENT_WHEELHOUSE_TAR:?set TENCENT_AGS_AGENT_WHEELHOUSE_TAR to a Python 3.10 wheelhouse tar.gz}"
  export TENCENT_AGS_TEMPLATE
  export TENCENT_AGS_IMAGE_REWRITE="${TENCENT_AGS_IMAGE_REWRITE:-}"
  export TENCENT_AGS_IMAGE_REGISTRY_TYPE
  export TENCENT_AGS_FORCE_HTTP1
  export TENCENT_AGS_AGENT_WHEELHOUSE_TAR
fi

TEMP_E2B_KEY_FILE=""
cleanup() {
  if [ -n "$TEMP_E2B_KEY_FILE" ] && [ -f "$TEMP_E2B_KEY_FILE" ]; then
    rm -f "$TEMP_E2B_KEY_FILE"
  fi
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [ -n "${E2B_API_KEY:-}" ]; then
  if [ -z "${E2B_API_KEY_FILE:-}" ]; then
    E2B_API_KEY_FILE="$(mktemp "${TMPDIR:-/tmp}/miles-miniswe-e2b-key.XXXXXX")"
    TEMP_E2B_KEY_FILE="$E2B_API_KEY_FILE"
  fi
  if [ ! -s "$E2B_API_KEY_FILE" ]; then
    printf '%s\n' "$E2B_API_KEY" > "$E2B_API_KEY_FILE"
    chmod 600 "$E2B_API_KEY_FILE"
  fi
  export E2B_API_KEY_FILE
  unset E2B_API_KEY
fi

if [ -z "${E2B_API_KEY:-}" ] && [ -z "${E2B_API_KEY_FILE:-}" ]; then
  echo "[run] set E2B_API_KEY or E2B_API_KEY_FILE for the E2B-compatible sandbox" >&2
  exit 2
fi

export HARBOR_ENV_TYPE
export HARBOR_TASKS_DIR
export HARBOR_TRIALS_DIR
export AGENT_MODEL_NAME=model
export AGENT_TIMEOUT=5400
export AGENT_TRIAL_TIMEOUT=7200
export AGENT_MAX_INPUT_TOKENS=28672
export AGENT_MAX_OUTPUT_TOKENS=4096
export HARBOR_MAX_SEQ_LEN=32768
export MILES_HOST_IP
export MILES_DUMP_DIR
export EXPECTED_LOGICAL_ROLLOUTS
export PYTHONPATH="$WS:$MILES_SOURCE_DIR:$MILES_SOURCE_DIR/examples/experimental/harbor:$MILES_SOURCE_DIR/examples/swe-agent-harbor-docker${HARBOR_SOURCE_DIR:+:$HARBOR_SOURCE_DIR/src}${PYTHONPATH:+:$PYTHONPATH}"

skip_prepare_args=()
if [ "${MILES_SKIP_PREPARE:-0}" = "1" ]; then
  skip_prepare_args=(--skip-prepare)
fi

mkdir -p "$MILES_DUMP_DIR" "$HARBOR_TRIALS_DIR"
driver_log="$AUTORLBENCH_RESULTS/experiment_logs/driver_$(date -u +%Y%m%d_%H%M%S).log"

set +e
python3 "$WS/run_miles.py" \
  --mode "$RUN_MODE" \
  --sandbox-backend "$SANDBOX_BACKEND" \
  --num-gpus-per-node 8 \
  --megatron-path "$MEGATRON_PATH" \
  --base-dir "$MILES_MODEL_BASE" \
  --model-name "$MODEL_NAME" \
  --hf-checkpoint "$MODEL_PATH" \
  --prompt-data "$MILES_PROMPT_DATA" \
  --save-traces-dir "$MILES_DUMP_DIR" \
  --harbor-tasks-dir "$HARBOR_TASKS_DIR" \
  --harbor-trials-dir "$HARBOR_TRIALS_DIR" \
  "${skip_prepare_args[@]}" \
  2>&1 | tee "$driver_log"
STATUS="${PIPESTATUS[0]}"
set -e

python3 "$WS/summarize_run.py"
SUMMARY_STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  exit "$STATUS"
fi
exit "$SUMMARY_STATUS"
