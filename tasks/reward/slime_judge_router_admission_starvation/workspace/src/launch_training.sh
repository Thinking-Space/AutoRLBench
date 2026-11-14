#!/usr/bin/env bash
#
# GUI Actor RL — cluster single-node variant of run_gui_actor_rl_qwen3vl_4b.sh.
#
# Same training recipe (Qwen3-VL-4B actor, GRPO + KL, frozen judge reward), with
# the bare-metal assumptions removed:
#   * no /workspace paths — everything derives from this checkout or env;
#   * first-party deps (mm_agents, evaluation_harness, ...) resolve from
#     external/pkgs, Megatron from MEGATRON_DIR (persisted copy, not /root);
#   * env services come from gui_actor_rl/scripts/start_env_host_ags.sh (AGS
#     sandboxes) and may run in this same pod;
#   * holo3 probe is OFF by default (no probe endpoint on this cluster);
#   * the multi-core mm preprocessing pool is enabled (the old single-node
#     script left it off and ran the HF processor on the event loop).
#
# 8-GPU layout (judge mode):  GPU 0-3 Megatron (TP=4) | GPU 4-6 SGLang rollout | GPU 7 judge
# 8-GPU layout (env mode):    GPU 0-3 Megatron (TP=4) | GPU 4-7 SGLang rollout
#
# Prerequisites (same pod or reachable host):
#   NUM_ENV_WORKERS=4 bash gui_actor_rl/scripts/start_env_host_ags.sh
#
# Smoke (2 tasks x 2 samples, 2 rollouts, 5 steps, env reward, no judge GPU):
#   SMOKE=1 GUI_REWARD_SOURCE=env NUM_ENV_WORKERS=4 bash gui_actor_rl/scripts/run_gui_actor_rl_qwen3vl_4b_cluster.sh
# Then judge mode:
#   SMOKE=1 NUM_ENV_WORKERS=4 bash gui_actor_rl/scripts/run_gui_actor_rl_qwen3vl_4b_cluster.sh

set -euo pipefail

# ── paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="${GUI_LAUNCHER_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SLIME_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# New-job-only engineering profile; the original formal launcher stays unchanged.
source "${SCRIPT_DIR}/cuagym_speed.env"
MODE_ARGS=()
TRAIN_ENTRY=train.py
case "${GUI_EXECUTION_MODE}" in
   dedicated) ;;
   async)
      TRAIN_ENTRY=train_async.py
      # One batch of prefetch: bound policy lag to one optimizer update. PPO's
      # denominator must use the behavior policy that actually sampled it.
      MODE_ARGS=(--update-weights-interval 1 --use-rollout-logprobs)
      ;;
   colocated)
      MODE_ARGS=(--colocate)
      if ! awk -v x="${SGLANG_MEM_FRACTION}" 'BEGIN { exit !(x > 0 && x <= 0.75) }'; then
         echo "colocated SGLANG_MEM_FRACTION must be in (0, 0.75]" >&2
         exit 1
      fi
      ;;
   *) echo "GUI_EXECUTION_MODE must be dedicated, colocated or async" >&2; exit 1 ;;
esac
if [[ -z "${GUI_SGLANG_TOKENIZER_WORKERS:-}" ]]; then
   if [[ "${GUI_EXECUTION_MODE}" == "dedicated" ]]; then
      export GUI_SGLANG_TOKENIZER_WORKERS=2
   else
      export GUI_SGLANG_TOKENIZER_WORKERS=1
   fi
fi
if [[ ! "${GUI_MISMATCH_METRICS_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
   echo "GUI_MISMATCH_METRICS_INTERVAL must be a positive integer" >&2
   exit 1
fi
PKGS_DIR="${SLIME_DIR}/external/pkgs"
MEGATRON_DIR="${MEGATRON_DIR:-/workspace/local/code/Megatron-LM}"
SAVE_BASE="${SAVE_BASE:-/workspace/local/gui_actor_ckpts}"

# Actor policy to train (HF format Qwen3-VL-4B-Thinking).
ACTOR_HF_CHECKPOINT="${ACTOR_HF_CHECKPOINT:-/workspace/local/LLM_models/Qwen3-VL-4B-Thinking}"
# Frozen judge: GTFix final GUIRL judge (archive/ckpts MANIFEST.txt).
JUDGE_HF_CHECKPOINT="${JUDGE_HF_CHECKPOINT:-/workspace/assets/judge-checkpoint/rollout_17}"

RUN_TAG="${RUN_TAG:-gui_actor_qwen3vl4b_judge_reward}"
GUI_ACTOR_RUN_DIR="${GUI_ACTOR_RUN_DIR:-/workspace/local/gui_actor_runs/${RUN_TAG}}"
TRAIN_LOG="${GUI_ACTOR_TRAIN_LOG:-${GUI_ACTOR_RUN_DIR}/train.log}"
RUNTIME_CONFIG="${GUI_ACTOR_RUNTIME_CONFIG:-${GUI_ACTOR_RUN_DIR}/runtime_config.yaml}"
MEGATRON_SAVE_DIR="${MEGATRON_SAVE_DIR:-${SAVE_BASE}/${RUN_TAG}}"
HF_SAVE_DIR="${HF_SAVE_DIR:-${SAVE_BASE}/${RUN_TAG}_hf}"
JUDGE_LOG="${GUI_ACTOR_RUN_DIR}/judge_sglang.log"
mkdir -p "${GUI_ACTOR_RUN_DIR}/logs"
[ "${NO_SAVE:-0}" = "1" ] || mkdir -p "${HF_SAVE_DIR}"
: > "${TRAIN_LOG}"

# ── reward source ──────────────────────────────────────────────────
GUI_REWARD_SOURCE="${GUI_REWARD_SOURCE:-judge}"
GUI_REWARD_SOURCE="${GUI_REWARD_SOURCE,,}"
case "${GUI_REWARD_SOURCE}" in
   judge|env|zerogui) ;;
   *)
      echo "ERROR: GUI_REWARD_SOURCE must be judge, env or zerogui; got: ${GUI_REWARD_SOURCE}"
      exit 1
      ;;
esac

# External judge: the reward services run on another rank/pod (start_reward_services.sh),
# so this role must not start a local judge and must not reserve a GPU for one.
GUI_JUDGE_EXTERNAL="${GUI_JUDGE_EXTERNAL:-0}"

# ── GPU layout ─────────────────────────────────────────────────────
NUM_GPUS_TOTAL="${NUM_GPUS_TOTAL:-8}"
JUDGE_GPU="${JUDGE_GPU:-$(( NUM_GPUS_TOTAL - 1 ))}"
ACTOR_NUM_GPUS="${ACTOR_NUM_GPUS:-4}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
for _layout_var in ACTOR_NUM_NODES ACTOR_NUM_GPUS NUM_GPUS_TOTAL; do
   if [[ ! "${!_layout_var}" =~ ^[1-9][0-9]*$ ]]; then
      echo "${_layout_var} must be a positive integer" >&2; exit 1
   fi
done
# ACTOR_NUM_GPUS means GPUs per training node, as in the original launcher.
ACTOR_TOTAL_GPUS=$(( ACTOR_NUM_NODES * ACTOR_NUM_GPUS ))
if (( ACTOR_NUM_NODES > 1 )) && [[ "${GUI_RAY_EXTERNAL:-0}" != "1" ]]; then
   echo "Multi-node training requires the supervisor's external Ray cluster" >&2; exit 1
fi
if (( ACTOR_NUM_GPUS >= NUM_GPUS_TOTAL / 2 + 1 )) && (( NUM_GPUS_TOTAL < 8 )); then
   echo "NOTE: on ${NUM_GPUS_TOTAL} GPUs set ACTOR_NUM_GPUS (default 4) so that rollout (+judge) GPUs remain, e.g. ACTOR_NUM_GPUS=2"
fi
if [ "${GUI_REWARD_SOURCE}" = "judge" ] && [ "${GUI_JUDGE_EXTERNAL}" != "1" ]; then
   DEFAULT_ROLLOUT_NUM_GPUS=$(( NUM_GPUS_TOTAL - ACTOR_TOTAL_GPUS - 1 ))
   JUDGE_NUM_GPUS=1
else
   DEFAULT_ROLLOUT_NUM_GPUS=$(( NUM_GPUS_TOTAL - ACTOR_TOTAL_GPUS ))
   JUDGE_NUM_GPUS=0
fi
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${DEFAULT_ROLLOUT_NUM_GPUS}}"
RAY_NUM_GPUS=$(( ACTOR_TOTAL_GPUS + ROLLOUT_NUM_GPUS ))
if [[ "${GUI_EXECUTION_MODE}" == "colocated" ]]; then
   ROLLOUT_NUM_GPUS="${ACTOR_TOTAL_GPUS}"
   RAY_NUM_GPUS="${ACTOR_TOTAL_GPUS}"
fi
# 4B fits on one GPU: TP=1 (pure DP) avoids per-layer TP all-reduces; the distributed
# optimizer shards Adam state across DP instead.
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
GUI_RAY_EXTERNAL="${GUI_RAY_EXTERNAL:-0}"

if (( ROLLOUT_NUM_GPUS < 1 )); then
   echo "ERROR: ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}; need NUM_GPUS_TOTAL > ACTOR_NUM_GPUS (+1 judge GPU in judge mode)"
   exit 1
fi
if (( RAY_NUM_GPUS + JUDGE_NUM_GPUS > NUM_GPUS_TOTAL )); then
   echo "ERROR: ACTOR_TOTAL_GPUS(${ACTOR_TOTAL_GPUS}) + ROLLOUT_NUM_GPUS(${ROLLOUT_NUM_GPUS}) + JUDGE_NUM_GPUS(${JUDGE_NUM_GPUS}) exceeds NUM_GPUS_TOTAL(${NUM_GPUS_TOTAL})"
   exit 1
fi
if (( TENSOR_MODEL_PARALLEL_SIZE < 1 || ACTOR_NUM_GPUS % TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
   echo "ERROR: TP(${TENSOR_MODEL_PARALLEL_SIZE}) must divide ACTOR_NUM_GPUS(${ACTOR_NUM_GPUS})"
   exit 1
fi
if [ "${GUI_REWARD_SOURCE}" = "judge" ] && [ "${GUI_JUDGE_EXTERNAL}" != "1" ] && (( JUDGE_GPU >= NUM_GPUS_TOTAL )); then
   echo "ERROR: JUDGE_GPU(${JUDGE_GPU}) must be < NUM_GPUS_TOTAL(${NUM_GPUS_TOTAL})"
   exit 1
fi

# ── judge server ───────────────────────────────────────────────────
JUDGE_PORT="${JUDGE_PORT:-30100}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-guirl-judge}"
JUDGE_MEM_FRACTION="${JUDGE_MEM_FRACTION:-0.85}"
# With GUI_JUDGE_EXTERNAL=1 this is injected by the launcher (another pod's IP:port).
JUDGE_BASE_URL="${GUI_JUDGE_BASE_URL:-http://host_0:${JUDGE_PORT}/v1}"

# ── env services (start_env_host_ags.sh; must already be running) ──
ENV_MASTER_HOST="${ENV_MASTER_HOST:-host_0}"
ENV_MASTER_PORT="${ENV_MASTER_PORT:-9900}"
GUI_ENV_MASTER_URL="http://${ENV_MASTER_HOST}:${ENV_MASTER_PORT}"
NUM_ENV_WORKERS="${NUM_ENV_WORKERS:-4}"
GUI_EXPECTED_ENV_WORKERS="${GUI_EXPECTED_ENV_WORKERS:-${NUM_ENV_WORKERS}}"
HOST_WAIT_TIMEOUT_SECONDS="${HOST_WAIT_TIMEOUT_SECONDS:-1800}"

ROLLOUT_MASTER_PORT="${ROLLOUT_MASTER_PORT:-9910}"
GUI_ROLLOUT_MASTER_URL="${GUI_ROLLOUT_MASTER_URL:-http://${ENV_MASTER_HOST}:${ROLLOUT_MASTER_PORT}}"

# ── holo3 probe: OFF by default here (no holo3 endpoint on this cluster).
# With the probe off, a judge verdict of "uncertain" drops the sample.
GUI_PROBE_ENABLE="${GUI_PROBE_ENABLE:-false}"
GUI_PROBE_BASE_URL="${GUI_PROBE_BASE_URL:-http://host_0:30000/v1}"
GUI_PROBE_MODEL_NAME="${GUI_PROBE_MODEL_NAME:-holo3-1-35b-a3b}"
GUI_PROBE_HOSTPORT="$(echo "${GUI_PROBE_BASE_URL}" | sed -E 's#^https?://##; s#/.*$##')"
GUI_PROBE_HOST="${GUI_PROBE_HOSTPORT%%:*}"

# ── task data ──────────────────────────────────────────────────────
# GUI_TASK_SET picks the task source (gui_actor_rl/task_sets.py):
#   osworld — OSWorld evaluation_examples manifest (GUI_TEST_CONFIG_BASE_DIR / GUI_TRAIN_META_PATH)
#   cua_gym — materialized CUA Gym bundle (GUI_CUA_GYM_TASKS_DIR + GUI_CUA_GYM_MANIFEST, or
#             a directory scan narrowed by GUI_CUA_GYM_APP_TYPES / _DIFFICULTIES / _LIMIT)
export GUI_TASK_SET="${GUI_TASK_SET:-osworld}"
export GUI_CUA_GYM_TASKS_DIR="${GUI_CUA_GYM_TASKS_DIR:-/workspace/assets/cua_gym_tasks}"
export GUI_CUA_GYM_MANIFEST="${GUI_CUA_GYM_MANIFEST:-}"
export GUI_CUA_GYM_APP_TYPES="${GUI_CUA_GYM_APP_TYPES:-}"
export GUI_CUA_GYM_DIFFICULTIES="${GUI_CUA_GYM_DIFFICULTIES:-}"
export GUI_CUA_GYM_SKIP_MOCK_APPS="${GUI_CUA_GYM_SKIP_MOCK_APPS:-1}"
export GUI_CUA_GYM_LIMIT="${GUI_CUA_GYM_LIMIT:-}"
export GUI_TEST_CONFIG_BASE_DIR="${GUI_TEST_CONFIG_BASE_DIR:-/workspace/assets/evaluation_examples}"
if [ "${SMOKE:-0}" = "1" ]; then
   DEFAULT_TRAIN_META="${GUI_TEST_CONFIG_BASE_DIR}/test_smoke_nogdrive.json"
else
   DEFAULT_TRAIN_META="${GUI_TEST_CONFIG_BASE_DIR}/test_meta_noproxy_nogdrive_feasible_offline280_train_nomulti.json"
fi
export GUI_TRAIN_META_PATH="${GUI_TRAIN_META_PATH:-${DEFAULT_TRAIN_META}}"
export GUI_TASK_SHUFFLE_SEED="${GUI_TASK_SHUFFLE_SEED:-42}"
export GUI_EVAL_META_PATH="${GUI_EVAL_META_PATH:-}"
export GUI_RESULT_DIR="${GUI_RESULT_DIR:-${GUI_ACTOR_RUN_DIR}/results}"
export GUI_POOL_MAX_ENVS="${GUI_POOL_MAX_ENVS:-${NUM_ENV_WORKERS}}"
export GUI_TRAJECTORY_CONCURRENCY="${GUI_TRAJECTORY_CONCURRENCY:-${NUM_ENV_WORKERS}}"
export GUI_ACTION_SPACE="${GUI_ACTION_SPACE:-pyautogui}"
export GUI_OBSERVATION_TYPE="${GUI_OBSERVATION_TYPE:-screenshot}"
export GUI_COORDINATE_TYPE="${GUI_COORDINATE_TYPE:-relative}"

GUI_TIMING_LOG_ENABLE="${GUI_TIMING_LOG_ENABLE:-true}"
GUI_TIMING_LOG_DIR="${GUI_TIMING_LOG_DIR:-${GUI_ACTOR_RUN_DIR}/logs/rollout_timing}"
mkdir -p "${GUI_RESULT_DIR}"

# ── preflight (fail fast, before any GPU is touched) ───────────────
PREFLIGHT_PATHS=("${ACTOR_HF_CHECKPOINT}/config.json" "${MEGATRON_DIR}/megatron/core/__init__.py" "${PKGS_DIR}/mm_agents" "${SLIME_DIR}/${TRAIN_ENTRY}")
case "${GUI_TASK_SET}" in
   osworld) PREFLIGHT_PATHS+=("${GUI_TRAIN_META_PATH}" "${GUI_TEST_CONFIG_BASE_DIR}/examples") ;;
   cua_gym) PREFLIGHT_PATHS+=("${GUI_CUA_GYM_TASKS_DIR}" ${GUI_CUA_GYM_MANIFEST:+"${GUI_CUA_GYM_MANIFEST}"}) ;;
   combined) PREFLIGHT_PATHS+=("${GUI_COMBINED_MANIFEST}" "${GUI_SCALE_CUA_RUNTIME_DIR}/adapter.py") ;;
   *) echo "ERROR: unknown GUI_TASK_SET=${GUI_TASK_SET} (osworld|cua_gym)"; exit 1 ;;
esac
for p in "${PREFLIGHT_PATHS[@]}"; do
   if [ ! -e "${p}" ]; then
      echo "ERROR: missing ${p}"
      exit 1
   fi
done
if [ "${GUI_REWARD_SOURCE}" = "judge" ] && [ ! -e "${JUDGE_HF_CHECKPOINT}/config.json" ]; then
   echo "ERROR: JUDGE_HF_CHECKPOINT does not exist: ${JUDGE_HF_CHECKPOINT}"
   exit 1
fi
# An external judge is another pod's job; fail here rather than silently scoring 0.
if [ "${GUI_REWARD_SOURCE}" = "judge" ] && [ "${GUI_JUDGE_EXTERNAL}" = "1" ]; then
   echo -n "Checking external judge at ${JUDGE_BASE_URL} ..."
   if curl --noproxy '*' -sf "${JUDGE_BASE_URL}/models" >/dev/null 2>&1; then
      echo " OK"
   else
      echo " UNREACHABLE"
      echo "ERROR: external judge ${JUDGE_BASE_URL} is not reachable from this node"
      exit 1
   fi
fi
PYTHONPATH="${PKGS_DIR}:${SLIME_DIR}:${MEGATRON_DIR}" python3 - <<'PY' || exit 1
import importlib, sys
missing = []
for mod in ("tenacity", "zstandard", "sglang", "ray", "megatron.core", "megatron.bridge", "slime", "gui_actor_rl.generate_with_gui",
            "gui_actor_rl.gui_data_source", "gui_actor_rl.cua_gym_data_source", "gui_actor_rl.task_sets",
            "gui_actor_rl.agents.qwen3vl_agent", "guirl.judge.judge_prompts"):
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    print("ERROR: trainer python cannot import:\n  " + "\n  ".join(missing), file=sys.stderr)
    print("       (transport dependencies: pip install tenacity zstandard==0.25.0)", file=sys.stderr)
    sys.exit(1)
print("trainer import preflight OK")
PY

# ── training schedule ──────────────────────────────────────────────
if [ "${SMOKE:-0}" = "1" ]; then
   ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
   N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
   NUM_EPOCH=""
   NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
   GUI_MAX_STEPS="${GUI_MAX_STEPS:-5}"
   SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
else
   ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
   N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
   if [[ -n "${NUM_ROLLOUT:-}" ]]; then
      NUM_EPOCH=""
   else
      NUM_EPOCH="${NUM_EPOCH:-10}"
      NUM_ROLLOUT=""
   fi
   GUI_MAX_STEPS="${GUI_MAX_STEPS:-30}"
   # Count through the task-set switch so filters (cua_gym) and missing files (osworld)
   # are reflected; the trainer python already passed the import preflight above.
   TASK_COUNT="$(PYTHONPATH="${PKGS_DIR}:${SLIME_DIR}:${MEGATRON_DIR}" python3 - <<'PY'
import logging, types
logging.disable(logging.CRITICAL)
from gui_actor_rl.task_sets import task_set_data_source
print(len(task_set_data_source(types.SimpleNamespace(n_samples_per_prompt=1, save=None, load=None))))
PY
)"
   STEPS_PER_EPOCH=$(( TASK_COUNT / ROLLOUT_BATCH_SIZE ))
   if [ "${STEPS_PER_EPOCH}" -lt 1 ]; then
      echo "ERROR: STEPS_PER_EPOCH=${STEPS_PER_EPOCH} (tasks=${TASK_COUNT}, batch=${ROLLOUT_BATCH_SIZE})"
      exit 1
   fi
   SAVE_INTERVAL="${SAVE_INTERVAL:-${STEPS_PER_EPOCH}}"
fi

# Each in-flight trajectory holds one desktop; below batch x n the rollout runs in waves.
TRAJECTORIES_PER_ROLLOUT=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))
if [ "${GUI_TRAJECTORY_CONCURRENCY}" -lt "${TRAJECTORIES_PER_ROLLOUT}" ]; then
   echo "WARNING: GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY} < batch x n = ${TRAJECTORIES_PER_ROLLOUT};" \
        "rollout will run in waves. Start >= ${TRAJECTORIES_PER_ROLLOUT} env workers (NUM_ENV_WORKERS)."
fi

LR="${LR:-1e-6}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-0}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
ADAM_BETA2="${ADAM_BETA2:-0.98}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
REF_LOAD="${REF_LOAD:-${ACTOR_HF_CHECKPOINT}}"

# Every intra-cluster HTTP hop (slime -> SGLang server, slime -> sgl-router, router -> workers,
# rollout -> router) is addressed by this node's pod IP. The inherited http_proxy is a whitelist
# egress proxy that answers 502 for such targets, so the pod IP, the private ranges and in-cluster
# service suffixes must bypass it — otherwise SGLangEngine.init() spins forever on 502 health probes
# and RolloutManager.__init__ never returns (observed on jobs 1234606 / 1234884).
NODE_IP="$( { hostname -I 2>/dev/null || true; } | awk '{print $1}')"
export no_proxy="${no_proxy:+${no_proxy},}host_0,localhost,::1,${ENV_MASTER_HOST},${GUI_PROBE_HOST}${NODE_IP:+,${NODE_IP}},$(hostname),host_3/8,host_4/12,host_5/16,.svc,.cluster.local"
export NO_PROXY="${no_proxy}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

set -x
cd "${SLIME_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  GUI Actor RL (cluster) — ${GUI_REWARD_SOURCE}-reward training"
echo "════════════════════════════════════════════════════════════════"
echo "Actor policy:   ${ACTOR_HF_CHECKPOINT}"
echo "Reward source:  ${GUI_REWARD_SOURCE}  (probe: ${GUI_PROBE_ENABLE})"
if [ "${GUI_REWARD_SOURCE}" = "judge" ]; then
   if [ "${GUI_JUDGE_EXTERNAL}" = "1" ]; then
      echo "Frozen judge:   ${JUDGE_HF_CHECKPOINT}  (external: ${JUDGE_BASE_URL})"
   else
      echo "Frozen judge:   ${JUDGE_HF_CHECKPOINT}  (GPU ${JUDGE_GPU}, :${JUDGE_PORT})"
   fi
fi
echo "Env master:     ${GUI_ENV_MASTER_URL}  (expect ${GUI_EXPECTED_ENV_WORKERS} workers)"
echo "Train meta:     ${GUI_TRAIN_META_PATH}"
echo "Task set:       ${GUI_TASK_SET}$([ "${GUI_TASK_SET}" = "cua_gym" ] && echo "  (${GUI_CUA_GYM_TASKS_DIR}; manifest=${GUI_CUA_GYM_MANIFEST:-scan})")"
echo "Megatron:       ${MEGATRON_DIR}"
echo "Layout:         train=${ACTOR_TOTAL_GPUS} (${ACTOR_NUM_NODES} nodes x ${ACTOR_NUM_GPUS}, TP=${TENSOR_MODEL_PARALLEL_SIZE}) rollout=${ROLLOUT_NUM_GPUS} judge=${JUDGE_NUM_GPUS}"
echo "Schedule:       batch=${ROLLOUT_BATCH_SIZE} x n=${N_SAMPLES_PER_PROMPT}, num_epoch=${NUM_EPOCH:-n/a} num_rollout=${NUM_ROLLOUT:-auto}, max_steps=${GUI_MAX_STEPS}"
echo "Optimizer:      lr=${LR}, warmup_iters=${LR_WARMUP_ITERS}, decay=constant, clip=${EPS_CLIP}/${EPS_CLIP_HIGH}, kl=${KL_LOSS_COEF}"
echo "════════════════════════════════════════════════════════════════"

# ── stop leftovers (ONLY this checkout's processes; the pod may be shared) ──
JUDGE_PID=""
cleanup() {
   set +e
   if [[ -n "${JUDGE_PID:-}" ]] && kill -0 "${JUDGE_PID}" 2>/dev/null; then
      kill "${JUDGE_PID}" || true
   fi
}
trap cleanup EXIT INT TERM

if [[ "${GUI_RAY_EXTERNAL}" != "1" ]]; then
   if [ "${GUI_JUDGE_EXTERNAL}" != "1" ]; then
      pkill -9 -f "sglang.launch_server .*--port ${JUDGE_PORT}" || true
   fi
   pkill -9 -f "${SLIME_DIR}/train.py" || true
   ray stop --force || true
   sleep 3
fi

# ── 1. frozen judge sglang server (before ray, on the excluded GPU) ─
if [ "${GUI_REWARD_SOURCE}" = "judge" ] && [ "${GUI_JUDGE_EXTERNAL}" != "1" ]; then
   CUDA_VISIBLE_DEVICES="${JUDGE_GPU}" python3 -m sglang.launch_server \
      --model-path "${JUDGE_HF_CHECKPOINT}" \
      --served-model-name "${JUDGE_MODEL_NAME}" \
      --host host_1 \
      --port "${JUDGE_PORT}" \
      --mem-fraction-static "${JUDGE_MEM_FRACTION}" \
      > "${JUDGE_LOG}" 2>&1 &
   JUDGE_PID=$!
   echo "Judge sglang PID=${JUDGE_PID}, log=${JUDGE_LOG}"
else
   echo "Skipping local frozen judge startup (reward source: ${GUI_REWARD_SOURCE})"
fi

if [ "${GUI_REWARD_SOURCE}" = "judge" ] && [ "${GUI_JUDGE_EXTERNAL}" != "1" ]; then
   echo -n "Waiting for judge server..."
   for i in $(seq 1 300); do
      if curl -sf "http://host_0:${JUDGE_PORT}/health" >/dev/null 2>&1; then
         echo " OK"
         break
      fi
      if ! kill -0 "${JUDGE_PID}" 2>/dev/null; then
         echo " FAILED (judge process died) — check ${JUDGE_LOG}"
         tail -50 "${JUDGE_LOG}" || true
         exit 1
      fi
      sleep 2
      if (( i == 300 )); then
         echo " TIMEOUT — check ${JUDGE_LOG}"
         exit 1
      fi
   done
fi

# ── 2. wait for env services ───────────────────────────────────────
python3 - "$ENV_MASTER_HOST" "$ENV_MASTER_PORT" "$GUI_EXPECTED_ENV_WORKERS" "$HOST_WAIT_TIMEOUT_SECONDS" <<'PY'
import json, sys, time, urllib.request
host, port, expected, timeout_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
deadline = time.time() + timeout_s
last_msg = ""
while time.time() < deadline:
    healthy = None
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/status", timeout=5) as resp:
            status = json.loads(resp.read().decode("utf-8"))
        healthy = int((status.get("master") or {}).get("healthy_workers") or 0)
        if healthy >= expected:
            print(f"Env master ready: healthy_workers={healthy}/{expected}", flush=True)
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as exc:
        healthy = f"error: {type(exc).__name__}"
    msg = f"waiting for env master {host}:{port}: healthy_workers={healthy}/{expected} (start gui_actor_rl/scripts/start_env_host_ags.sh)"
    if msg != last_msg:
        print(msg, flush=True)
        last_msg = msg
    time.sleep(5)
print(f"ERROR: env master not ready after {timeout_s}s", file=sys.stderr)
raise SystemExit(1)
PY

# ── 2b. rollout master / probe preflight (judge mode) ──────────────
if [ "${GUI_REWARD_SOURCE}" = "judge" ] && [ "${GUI_PROBE_ENABLE}" = "true" ]; then
   echo -n "Checking rollout master at ${GUI_ROLLOUT_MASTER_URL} ..."
   if curl -sf "${GUI_ROLLOUT_MASTER_URL}/healthz" >/dev/null 2>&1; then
      echo " OK"
   else
      echo " UNREACHABLE"
      echo "ERROR: rollout master (:${ROLLOUT_MASTER_PORT}) is required for the judge probe (GUI_PROBE_ENABLE=true)."
      exit 1
   fi
   echo -n "Checking probe endpoint at ${GUI_PROBE_BASE_URL} ..."
   if curl -sf "${GUI_PROBE_BASE_URL}/models" >/dev/null 2>&1; then
      echo " OK"
   else
      echo " UNREACHABLE"
      echo "WARNING: probe endpoint not reachable; uncertain verdicts will be dropped. Set GUI_PROBE_ENABLE=false to silence."
   fi
elif [ "${GUI_REWARD_SOURCE}" = "judge" ]; then
   echo "Probe disabled: judge 'uncertain' verdicts drop the sample (remove_sample=True)."
fi

# ── 3. runtime config (all gui_* args injected via custom-config) ──
cat > "$RUNTIME_CONFIG" <<EOF
gui_env_master_url: "${GUI_ENV_MASTER_URL}"
gui_reward_source: "${GUI_REWARD_SOURCE}"
gui_actor_train_node_ip: "${GUI_ACTOR_TRAIN_NODE_IP:-}"

# passive ZeroGUI judge (only when gui_reward_source == zerogui)
gui_zerogui_base_url: "${GUI_ZEROGUI_BASE_URL:-http://host_0:8030/v1}"
gui_zerogui_model_name: "${GUI_ZEROGUI_MODEL_NAME:-Qwen3-VL-4B-Thinking}"
gui_zerogui_api_key: "${GUI_ZEROGUI_API_KEY:-EMPTY}"
gui_zerogui_temperature: ${GUI_ZEROGUI_TEMPERATURE:-1.0}
gui_zerogui_top_p: ${GUI_ZEROGUI_TOP_P:-0.95}
gui_zerogui_max_tokens: ${GUI_ZEROGUI_MAX_TOKENS:-8192}
gui_zerogui_enable_thinking: ${GUI_ZEROGUI_ENABLE_THINKING:-true}
gui_zerogui_request_timeout: ${GUI_ZEROGUI_REQUEST_TIMEOUT:-1200}
gui_zerogui_max_screenshots: ${GUI_ZEROGUI_MAX_SCREENSHOTS:-15}

# frozen judge sampling — aligned to guirl TRAINING (temperature 1.0, 24576 tokens, thinking)
gui_judge_base_url: "${JUDGE_BASE_URL}"
gui_judge_model_name: "${JUDGE_MODEL_NAME}"
gui_judge_history_n: ${GUI_JUDGE_HISTORY_N:-3}
gui_judge_max_tokens: ${GUI_JUDGE_MAX_TOKENS:-24576}
gui_judge_temperature: ${GUI_JUDGE_TEMPERATURE:-1.0}
gui_judge_enable_thinking: ${GUI_JUDGE_ENABLE_THINKING:-true}
gui_judge_parse_max_retries: ${GUI_JUDGE_PARSE_MAX_RETRIES:-3}
gui_judge_concurrency: ${GUI_JUDGE_CONCURRENCY:-16}
gui_judge_max_rounds: ${GUI_JUDGE_MAX_ROUNDS:-3}

# holo3 probe (rollout master /run_probe_round); disabled by default on this cluster
gui_rollout_master_url: "${GUI_ROLLOUT_MASTER_URL}"
gui_probe_enable: ${GUI_PROBE_ENABLE}
gui_probe_agent_type: "${GUI_PROBE_AGENT_TYPE:-Holo3_Probe}"
gui_probe_base_url: "${GUI_PROBE_BASE_URL}"
gui_probe_model_name: "${GUI_PROBE_MODEL_NAME}"
gui_probe_api_key: "${GUI_PROBE_API_KEY:-EMPTY}"
gui_probe_max_steps: ${GUI_PROBE_MAX_STEPS:-15}
gui_probe_max_tokens: ${GUI_PROBE_MAX_TOKENS:-8192}
gui_probe_temperature: ${GUI_PROBE_TEMPERATURE:-0.0}
gui_probe_enable_thinking: ${GUI_PROBE_ENABLE_THINKING:-true}
gui_probe_reasoning_effort: "${GUI_PROBE_REASONING_EFFORT:-medium}"
gui_probe_history_n: ${GUI_PROBE_HISTORY_N:-4}
gui_probe_max_images: ${GUI_PROBE_MAX_IMAGES:-3}
gui_probe_max_trajectory_length: ${GUI_PROBE_MAX_TRAJECTORY_LENGTH:-15}
gui_probe_client_password: "${GUI_PROBE_CLIENT_PASSWORD:-password}"
gui_probe_sleep_after_execution: ${GUI_PROBE_SLEEP_AFTER_EXECUTION:-2.0}

# freeze the ViT blocks / the projector (vision merger + deepstack mergers); read by model_provider.py
freeze_vision_model: ${GUI_FREEZE_VISION:-false}
freeze_vision_projection: ${GUI_FREEZE_VISION_PROJECTION:-false}

gui_agent_class_path: "gui_actor_rl.agents.qwen3vl_agent.Qwen3VLAgentLocal"
gui_max_steps: ${GUI_MAX_STEPS}
gui_sleep_after_execution: ${GUI_SLEEP_AFTER_EXECUTION:-2.0}
gui_wait_after_reset: ${GUI_WAIT_AFTER_RESET:-30}
gui_max_image_history_length: ${GUI_MAX_IMAGE_HISTORY_LENGTH:-4}
gui_parse_max_retries: ${GUI_PARSE_MAX_RETRIES:-7}
gui_eval_max_steps: ${GUI_EVAL_MAX_STEPS:-${GUI_MAX_STEPS}}
gui_cua_gym_eval_manifest: "${GUI_CUA_GYM_EVAL_MANIFEST:-}"
gui_eval_sleep_after_execution: ${GUI_EVAL_SLEEP_AFTER_EXECUTION:-3.0}
gui_eval_wait_after_reset: ${GUI_EVAL_WAIT_AFTER_RESET:-30}

# "text" = only traj.jsonl / result.txt / judge.json per episode; "full" also writes step_N.png and
# trajectory.json (~100 MB per episode). Nothing in training reads the latter two.
gui_trajectory_artifacts: "${GUI_TRAJECTORY_ARTIFACTS:-full}"

# per-trajectory wall-clock budget (train only); must stay < AGS_SANDBOX_TIMEOUT
gui_trajectory_timeout_s: ${GUI_TRAJECTORY_TIMEOUT_S:-1800}
gui_eval_trajectory_timeout_s: ${GUI_EVAL_TRAJECTORY_TIMEOUT_S:-0}

dynamic_history: ${DYNAMIC_HISTORY:-true}

# multi-core mm preprocessing (gui_actor_rl/mm_preprocess.py): HF processor on
# Ray CPU actors, pixel_values as plasma views; per-image cache dedups the
# sliding-window screenshots. 0 = legacy inline path.
gui_mm_pool_size: ${GUI_MM_POOL_SIZE:-16}
gui_mm_per_image_cache: ${GUI_MM_PER_IMAGE_CACHE:-true}
# Ship each screenshot's pixel_values as uint8 + decode table (bit-exact, verified per image,
# falls back to float32 per image): 6.3 MB instead of 50 MB. Float32 made ~370 GB per rollout, overflowed
# the plasma store into the pod's 100 GiB ephemeral disk and got job 1323642 evicted.
gui_mm_pixel_u8: ${GUI_MM_PIXEL_U8:-true}

# gated rollout timing + per-task diagnostics (gui_actor_rl/timing.py)
gui_timing_log_enable: ${GUI_TIMING_LOG_ENABLE}
gui_timing_log_dir: "${GUI_TIMING_LOG_DIR}"
EOF

# Train/inference mismatch monitoring (GUI_MISMATCH_METRICS=1): slime recomputes the rollout
# log-probs of every sampled token with Megatron and reports mis_kl / mis_k3_kl / mis_*ppl* /
# train_rollout_logprob_abs_diff against the SGLang ones carried on Sample.rollout_log_probs.
# Monitoring only — the loss is untouched (no --use-tis). mis.py reads the tis/rs bounds from
# args even in this mode, so they must be present in the runtime config. mis.py only has means;
# gui_actor_rl.mismatch_metrics (rollout-data-postprocess hook) adds the exact max / quantiles /
# Pearson / worst-token dump under mismatch/* in wandb and ${GUI_ACTOR_RUN_DIR}/mismatch/.
MISMATCH_ARGS=()
if [ "${GUI_MISMATCH_METRICS:-0}" = "1" ]; then
   cat >> "$RUNTIME_CONFIG" <<EOF

# mismatch metrics (examples/train_infer_mismatch_helper/mis.py)
tis_lower_bound: 0.5
tis_upper_bound: 2.0
rs_lower_bound: null
rs_upper_bound: null

# exact (max / quantile / Pearson / per-bucket) diagnostics, gui_actor_rl/mismatch_metrics.py
gui_mismatch_dump_dir: "${GUI_ACTOR_RUN_DIR}/mismatch"
EOF
   MISMATCH_ARGS=(
      --get-mismatch-metrics
      --mismatch-metrics-interval "${GUI_MISMATCH_METRICS_INTERVAL}"
      --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
      --rollout-data-postprocess-path gui_actor_rl.mismatch_metrics.log_mismatch_metrics
   )
fi
echo "Runtime config written to ${RUNTIME_CONFIG}"

# ── 4. model args ──────────────────────────────────────────────────
# Text Qwen3-4B spec; with --megatron-to-hf-mode bridge the model is built from
# the HF config, these dims only feed validation (they match the VL text tower).
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(--hf-checkpoint "${ACTOR_HF_CHECKPOINT}")
# NO_SAVE=1 disables both periodic and real-data-epoch checkpoints.
if [ "${NO_SAVE:-0}" != "1" ]; then
   CKPT_ARGS+=(
      --save          "${MEGATRON_SAVE_DIR}"
      --save-hf       "${HF_SAVE_DIR}/rollout_{rollout_id}"
   )
   if [ "${SAVE_AT_DATA_EPOCH_END:-0}" = "1" ]; then
      CKPT_ARGS+=(--save-at-data-epoch-end)
   else
      CKPT_ARGS+=(--save-interval "${SAVE_INTERVAL}")
   fi
fi
if [[ "${GUI_KEEP_ALL_CHECKPOINTS:-0}" == "1" && -n "${SAVE_RETAIN_INTERVAL:-}" ]]; then
   echo 'All-checkpoint retention forbids SAVE_RETAIN_INTERVAL pruning' >&2
   exit 1
fi
# SAVE_RETAIN_INTERVAL (a multiple of SAVE_INTERVAL) keeps only the newest Megatron checkpoint:
# after a new iteration is fully written and the tracker file updated, Megatron deletes the
# previous one unless its iteration is a multiple of this value. The HF exports (--save-hf) are
# separate directories and are never deleted, so every save interval still leaves a HF copy.
if [[ -n "${SAVE_RETAIN_INTERVAL:-}" ]] && [ "${NO_SAVE:-0}" != "1" ]; then
   CKPT_ARGS+=(--save-retain-interval "${SAVE_RETAIN_INTERVAL}")
fi
if [ "${ENABLE_RESUME_LOAD:-0}" = "1" ]; then
   CKPT_ARGS+=(--load "${MEGATRON_SAVE_DIR}")
   if [ "${USE_CHECKPOINT_SCHEDULER:-0}" = "1" ]; then
      CKPT_ARGS+=(--use-checkpoint-opt-param-scheduler)
   fi
fi
if awk -v c="${KL_LOSS_COEF}" 'BEGIN { exit !(c + 0 > 0) }'; then
   CKPT_ARGS+=(--ref-load "${REF_LOAD}")
fi

ROLLOUT_ARGS=(
   --data-source-path              gui_actor_rl.task_sets.task_set_data_source
   --custom-generate-function-path gui_actor_rl.generate_with_gui.generate
   --custom-rm-path                gui_actor_rl.generate_with_gui.reward_func
   --custom-reward-post-process-path gui_actor_rl.generate_with_gui.reward_post_process
   --rollout-function-path         gui_actor_rl.generate_with_gui.gui_generate_rollout
   --eval-function-path            gui_actor_rl.generate_with_gui.gui_generate_rollout
   --custom-config-path            "${RUNTIME_CONFIG}"
   --reward-key                    score
   --eval-reward-key               acc
   --rollout-batch-size            "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt          "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len      "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
   --rollout-max-context-len       "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
   --rollout-temperature           "${ROLLOUT_TEMPERATURE:-1.0}"
   --rollout-top-p                 "${ROLLOUT_TOP_P:-0.9}"
   --num-steps-per-rollout         1
)
if [ -n "${NUM_EPOCH}" ]; then
   ROLLOUT_ARGS+=(--num-epoch "${NUM_EPOCH}")
else
   ROLLOUT_ARGS+=(--num-rollout "${NUM_ROLLOUT}")
fi
if [[ -n "${EVAL_INTERVAL:-}" ]]; then
   ROLLOUT_ARGS+=(--eval-interval "${EVAL_INTERVAL}" --eval-prompt-data cuagym "${SLIME_DIR}/gui_actor_rl/data/gui_eval_stub.jsonl"
      --eval-temperature "${EVAL_TEMPERATURE:-0.0}" --n-samples-per-eval-prompt 1)
fi
if [[ "${GUI_DAPO:-0}" == "1" ]]; then
   ROLLOUT_ARGS+=(--dynamic-sampling-filter-path gui_actor_rl.dapo.check_gui_reward_nonzero_std)
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip      "${EPS_CLIP}"
   --eps-clip-high "${EPS_CLIP_HIGH}"
   --entropy-coef  "${ENTROPY_COEF}"
   --kl-loss-type  "${KL_LOSS_TYPE}"
)
if awk -v c="${KL_LOSS_COEF}" 'BEGIN { exit !(c + 0 > 0) }'; then
   GRPO_ARGS+=(--use-kl-loss --kl-loss-coef "${KL_LOSS_COEF}")
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-warmup-iters "${LR_WARMUP_ITERS}"
   --lr-decay-style constant
   --weight-decay "${WEIGHT_DECAY:-0.1}"
   --adam-beta1 0.9
   --adam-beta2 "${ADAM_BETA2}"
)
# Optimizer CPU offload is on by default. Smoke 1321481 (TP=1, DP=2, no offload) OOMed in
# compute_log_prob: the resident Adam shard (~12 B/param / DP = 26.6 GB at DP=2) left
# 18 GB free, and the unsharded-vocab fp32 logits of one 32k-token pack need 18.4 GiB.
# OPTIMIZER_CPU_OFFLOAD=0 keeps Adam on GPU (TE FusedAdam); only use it after measuring
# headroom at the target DP. The update rule is the same fp32 AdamW either way (fp32
# master weights and moments), so only rounding order can differ.
# --use-precision-aware-optimizer is kept in both branches.
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-1}"
OPTIMIZER_ARGS+=(--use-precision-aware-optimizer)
if [ "${OPTIMIZER_CPU_OFFLOAD}" = "1" ]; then
   OPTIMIZER_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d)
elif [ "${GUI_OVERLAP_DP_COMM:-0}" = "1" ]; then
   # Opt-in until GPU-verified on this model: with overlapped grad reduce Megatron
   # issues each bucket's reduce once the per-param grad counts match those of the
   # first batch, and asserts at finish_grad_sync if a later batch differs.
   OPTIMIZER_ARGS+=(--overlap-grad-reduce --overlap-param-gather)
fi
if [[ -n "${LR_DECAY_ITERS:-}" ]]; then
   OPTIMIZER_ARGS+=(--lr-decay-iters "${LR_DECAY_ITERS}")
fi

# Sequence parallel only applies with TP>1.
SEQUENCE_PARALLEL_ARGS=()
(( TENSOR_MODEL_PARALLEL_SIZE > 1 )) && SEQUENCE_PARALLEL_ARGS=(--sequence-parallel)
# Activation recompute trades compute for memory and does not change the math.
# GUI_RECOMPUTE=full (default, smoke-verified) | selective | none.
case "${GUI_RECOMPUTE:-full}" in
   full) RECOMPUTE_ARGS=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1) ;;
   selective) RECOMPUTE_ARGS=(--recompute-granularity selective) ;;
   none) RECOMPUTE_ARGS=() ;;
   *) echo "ERROR: GUI_RECOMPUTE=${GUI_RECOMPUTE} (full|selective|none)"; exit 1 ;;
esac
PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
   "${SEQUENCE_PARALLEL_ARGS[@]}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --calculate-per-token-loss
   "${RECOMPUTE_ARGS[@]}"
   --megatron-to-hf-mode bridge
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-32768}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.75}"
   # Engine context = the budget qwen3vl_agent clamps max_new_tokens against.
   --sglang-context-length "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
   # cache_aware never crosses its load-imbalance thresholds at a few dozen in-flight
   # trajectories, so every request landed on one engine. consistent_hashing on the
   # X-SMG-Routing-Key header (= episode id, sent by qwen3vl_agent) keeps each
   # trajectory on one engine (prefix/ViT reuse) and spreads trajectories evenly.
   --router-policy "${ROLLOUT_ROUTER_POLICY:-consistent_hashing}"
)
if [[ "${ROLLOUT_ROUTER_POLICY:-consistent_hashing}" == "manual" ]]; then
   SGLANG_ARGS+=(--router-assignment-mode "${ROLLOUT_ROUTER_ASSIGNMENT_MODE:-min_load}")
fi
# A/B knobs. Omitted means use installed SGLang's defaults (CUDA Graph is
# already enabled; on H100 its chunked prefill defaults to 8192).
for _item in \
   "GUI_SGLANG_TOKENIZER_WORKERS:tokenizer-worker-num" \
   "GUI_SGLANG_CHUNKED_PREFILL:chunked-prefill-size" \
   "GUI_SGLANG_MAX_RUNNING_REQUESTS:max-running-requests" \
   "GUI_SGLANG_CUDA_GRAPH_MAX_BS:cuda-graph-max-bs"; do
   _env_name="${_item%%:*}"
   _flag="${_item#*:}"
   if [[ -n "${!_env_name:-}" ]]; then
      if [[ ! "${!_env_name}" =~ ^[1-9][0-9]*$ ]]; then
         echo "${_env_name} must be a positive integer" >&2; exit 1
      fi
      SGLANG_ARGS+=("--sglang-${_flag}" "${!_env_name}")
   fi
done
# Existing extra-argument hook for controlled experiments.
if [[ -n "${SMOKE_SGLANG_EXTRA_ARGS:-}" ]]; then
   read -r -a _SMOKE_EXTRA <<< "${SMOKE_SGLANG_EXTRA_ARGS}"
   SGLANG_ARGS+=("${_SMOKE_EXTRA[@]}")
fi

# W&B: explicit key, else ~/.netrc; nothing else is read.
WANDB_KEY_VALUE=""
if [[ "${TRACKING_BACKEND:-wandb}" == "wandb" ]]; then
WANDB_KEY_VALUE="${WANDB_KEY:-${WANDB_API_KEY:-}}"
if [ -z "${WANDB_KEY_VALUE}" ] && [ -f "${HOME}/.netrc" ]; then
   WANDB_KEY_VALUE="$(
      python3 - <<'PY'
from pathlib import Path
text = Path.home().joinpath(".netrc").read_text()
for block in text.split("machine "):
    if not block.strip():
        continue
    lines = block.strip().splitlines()
    if "wandb" not in lines[0]:
        continue
    for line in lines[1:]:
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] == "password":
            print(parts[1].strip())
            raise SystemExit
PY
   )"
fi
fi
if [ -n "${WANDB_KEY_VALUE}" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT:-gui_actor_rl}"
      --wandb-group "${WANDB_GROUP:-${RUN_TAG}}"
      --wandb-key "${WANDB_KEY_VALUE}"
      --wandb-dir "${GUI_ACTOR_RUN_DIR}/wandb"
   )
   echo "W&B enabled: project=${WANDB_PROJECT:-gui_actor_rl} group=${WANDB_GROUP:-${RUN_TAG}}"
else
   WANDB_ARGS=()
   echo "W&B disabled (set WANDB_API_KEY or run: wandb login)"
fi
if [[ "${TRACKING_BACKEND:-wandb}" == "swanlab" ]]; then
   python3 - <<'PY'
from importlib.metadata import version
from packaging.version import Version
if Version(version("swanlab")) < Version("0.10.0"):
    raise SystemExit("SwanLab SDK >= 0.10.0 is required")
PY
   GUI_SWANLAB_PROJECT="${GUI_SWANLAB_PROJECT:-${SWANLAB_PROJECT:-gui-actor-rl}}"
   unset SWANLAB_PROJECT
   WANDB_ARGS=(--use-swanlab --swanlab-project "${GUI_SWANLAB_PROJECT}"
      --swanlab-name "${SWANLAB_RUN_NAME:-${RUN_TAG}}" --swanlab-mode "${SWANLAB_MODE:-online}"
      --swanlab-log-dir "${GUI_ACTOR_RUN_DIR}/swanlab"
      --swanlab-api-key-file "${SWANLAB_API_KEY_FILE:-/workspace/local/.env/swanlab}")
fi

# ── 5. ray ─────────────────────────────────────────────────────────
NUM_GPUS_TOTAL="${NUM_GPUS_TOTAL:-${RAY_NUM_GPUS}}"
export NUM_GPUS_TOTAL

NVLINK_COUNT=$( { nvidia-smi topo -m 2>/dev/null || true; } | { grep -o 'NV[0-9][0-9]*' || true; } | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

if [ "${GUI_REWARD_SOURCE}" = "judge" ] && [ "${GUI_JUDGE_EXTERNAL}" != "1" ]; then
   RAY_GPU_IDS="$(python3 -c "print(','.join(str(i) for i in range(${NUM_GPUS_TOTAL}) if i != ${JUDGE_GPU}))")"
else
   RAY_GPU_IDS="$(python3 -c "print(','.join(str(i) for i in range(${NUM_GPUS_TOTAL})))")"
fi

if [[ "${GUI_RAY_EXTERNAL}" != "1" ]]; then
CUDA_VISIBLE_DEVICES="${RAY_GPU_IDS}" ray start --head \
   --node-ip-address host_0 \
   --num-gpus "${RAY_NUM_GPUS}" \
   --disable-usage-stats \
   --dashboard-host=host_1 \
   --dashboard-port=8265
fi

RAY_DASHBOARD_READY=0
for i in $(seq 1 30); do
   if curl --noproxy '*' -sf http://host_0:8265/api/jobs/ >/dev/null 2>&1; then
      RAY_DASHBOARD_READY=1
      break
   fi
   sleep 2
done
if [[ "${RAY_DASHBOARD_READY}" != "1" ]]; then
   echo "Ray dashboard failed its readiness check" >&2
   exit 1
fi

# CPU threads: os.cpu_count() reports the whole host (192) while the cgroup grants far
# fewer CPUs, and torch/OMP size their pools from the former (HF processor measured
# 7x slower at 96 threads than at 8 on an 8-CPU quota). Default to the quota, max 8.
CGROUP_CPUS="$(awk '$1 != "max" { c = int($1 / $2); print (c < 1 ? 1 : c) }' /sys/fs/cgroup/cpu.max 2>/dev/null || true)"
DEFAULT_CPU_THREADS="${CGROUP_CPUS:-8}"
(( DEFAULT_CPU_THREADS > 8 )) && DEFAULT_CPU_THREADS=8
# ViT embedding cache per engine (GPU memory, LRU). The 100 MB SGLang default holds ~2
# screenshots (~42 MB each with deepstack), so every step re-encoded the whole image
# history. 4 GB covers ~16 trajectories x 5 images per engine. Entries are dropped on
# every weight-update flush (slime/backends/sglang_utils/sglang_plugins.py).
SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-4096}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PKGS_DIR}:${SLIME_DIR}:${MEGATRON_DIR}\",
    \"AGENTPLAT_PATH\": \"${PKGS_DIR}\",
    \"GUI_REWARD_AGENT_PATH\": \"${PKGS_DIR}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"PYTORCH_ALLOC_CONF\": \"${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}\",
    \"GUI_ENV_MASTER_URL\": \"${GUI_ENV_MASTER_URL}\",
    \"GUI_POOL_MAX_ENVS\": \"${GUI_POOL_MAX_ENVS}\",
    \"GUI_TRAJECTORY_CONCURRENCY\": \"${GUI_TRAJECTORY_CONCURRENCY}\",
    \"GUI_RESULT_DIR\": \"${GUI_RESULT_DIR}\",
    \"GUI_ACTION_SPACE\": \"${GUI_ACTION_SPACE}\",
    \"GUI_OBSERVATION_TYPE\": \"${GUI_OBSERVATION_TYPE}\",
    \"GUI_COORDINATE_TYPE\": \"${GUI_COORDINATE_TYPE}\",
    \"GUI_TEST_CONFIG_BASE_DIR\": \"${GUI_TEST_CONFIG_BASE_DIR}\",
    \"GUI_TRAIN_META_PATH\": \"${GUI_TRAIN_META_PATH}\",
    \"GUI_TASK_SET\": \"${GUI_TASK_SET}\",
    \"GUI_COMBINED_MANIFEST\": \"${GUI_COMBINED_MANIFEST:-}\",
    \"GUI_SCALE_CUA_RUNTIME_DIR\": \"${GUI_SCALE_CUA_RUNTIME_DIR:-}\",
    \"GUI_SCALE_SHARED_RUNTIME\": \"${SLIME_DIR}\",
    \"GUI_CUA_GYM_TASKS_DIR\": \"${GUI_CUA_GYM_TASKS_DIR}\",
    \"GUI_CUA_GYM_MANIFEST\": \"${GUI_CUA_GYM_MANIFEST}\",
    \"GUI_CUA_GYM_APP_TYPES\": \"${GUI_CUA_GYM_APP_TYPES}\",
    \"GUI_CUA_GYM_DIFFICULTIES\": \"${GUI_CUA_GYM_DIFFICULTIES}\",
    \"GUI_CUA_GYM_SKIP_MOCK_APPS\": \"${GUI_CUA_GYM_SKIP_MOCK_APPS}\",
    \"GUI_CUA_GYM_LIMIT\": \"${GUI_CUA_GYM_LIMIT}\",
    \"GUI_TASK_SHUFFLE_SEED\": \"${GUI_TASK_SHUFFLE_SEED}\",
    \"GUI_ENV_WORKER_SLEEPS_AFTER_EXECUTION\": \"${GUI_ENV_WORKER_SLEEPS_AFTER_EXECUTION:-1}\",
    \"OMP_NUM_THREADS\": \"${OMP_NUM_THREADS:-${DEFAULT_CPU_THREADS}}\",
    \"MKL_NUM_THREADS\": \"${MKL_NUM_THREADS:-${DEFAULT_CPU_THREADS}}\",
    \"SGLANG_VLM_CACHE_SIZE_MB\": \"${SGLANG_VLM_CACHE_SIZE_MB}\",
    \"RAY_DEDUP_LOGS\": \"${RAY_DEDUP_LOGS:-1}\",
    \"SLIME_SGLANG_MM_U8_TRANSPORT\": \"${SLIME_SGLANG_MM_U8_TRANSPORT:-0}\",
    \"SLIME_HTTP_FAIL_FAST_4XX\": \"${SLIME_HTTP_FAIL_FAST_4XX:-0}\",
    \"SLIME_HTTP_FAST_JSON\": \"${SLIME_HTTP_FAST_JSON:-0}\",
    \"SLIME_HTTP_PROFILE\": \"${SLIME_HTTP_PROFILE:-0}\",
    \"GUI_TOKENIZE_OFFLOAD\": \"${GUI_TOKENIZE_OFFLOAD:-1}\",
    \"GUI_BATCH_NAVIGATION_KEYS\": \"${GUI_BATCH_NAVIGATION_KEYS:-0}\",
    \"GUI_REQUIRE_COMPLETE_BATCH\": \"${GUI_REQUIRE_COMPLETE_BATCH:-0}\",
    \"GUI_DAPO\": \"${GUI_DAPO:-0}\",
    \"GUI_DAPO_STD_MIN\": \"${GUI_DAPO_STD_MIN:-0.000001}\",
    \"GUI_DAPO_MAX_GROUPS\": \"${GUI_DAPO_MAX_GROUPS:-192}\",
    \"GUI_DAPO_TIMEOUT_S\": \"${GUI_DAPO_TIMEOUT_S:-6000}\",
    \"GUI_DAPO_LOOKAHEAD_GROUPS\": \"${GUI_DAPO_LOOKAHEAD_GROUPS:-0}\",
    \"GUI_DAPO_CLEANUP_TIMEOUT_S\": \"${GUI_DAPO_CLEANUP_TIMEOUT_S:-180}\",
    \"GUI_TYPEWRITE_INTERVAL\": \"${GUI_TYPEWRITE_INTERVAL:-0.03}\",
    \"GUI_CUA_NATIVE_TOOLS\": \"${GUI_CUA_NATIVE_TOOLS:-0}\",
    \"GUI_CUA_VLC_STATE_CAPTURE\": \"${GUI_CUA_VLC_STATE_CAPTURE:-0}\",
    \"GUI_CUA_TASK_OVERRIDES\": \"${GUI_CUA_TASK_OVERRIDES:-}\",
    \"GUI_CUA_SETUP_ASSETS_DIR\": \"${GUI_CUA_SETUP_ASSETS_DIR:-/workspace/local/data/cua_setup_assets_20261026}\",
    \"GUI_CUA_PYTHON_ASSETS_DIR\": \"${GUI_CUA_PYTHON_ASSETS_DIR:-/workspace/local/data/cua_python_wheels_20261026_offline}\",
    \"GUI_CUA_REQUIRE_OFFLINE_PYTHON\": \"${GUI_CUA_REQUIRE_OFFLINE_PYTHON:-1}\",
    \"GUI_MM_PIXEL_COMPRESS\": \"${GUI_MM_PIXEL_COMPRESS:-1}\",
    \"SLIME_IMAGE_REFS\": \"${SLIME_IMAGE_REFS:-0}\",
    \"SLIME_IMAGE_RAW_CACHE_MB\": \"${SLIME_IMAGE_RAW_CACHE_MB:-256}\",
    \"SLIME_IMAGE_PREPROCESS_CACHE\": \"${SLIME_IMAGE_PREPROCESS_CACHE:-0}\",
    \"SLIME_IMAGE_PREPROCESS_CACHE_MB\": \"${SLIME_IMAGE_PREPROCESS_CACHE_MB:-2048}\",
    \"SLIME_IMAGE_PREPROCESS_COMPRESS\": \"${SLIME_IMAGE_PREPROCESS_COMPRESS:-1}\",
    \"SLIME_SGLANG_CONTROL_TIMEOUT_SECS\": \"${SLIME_SGLANG_CONTROL_TIMEOUT_SECS:-300}\",
    \"GUI_LOG_RESPONSE_PREVIEW_CHARS\": \"${GUI_LOG_RESPONSE_PREVIEW_CHARS:-0}\",
    \"no_proxy\": \"${no_proxy}\",
    \"NO_PROXY\": \"${NO_PROXY}\"
  }
}"

# ── 6. submit ──────────────────────────────────────────────────────
RAY_JOB_SUBMISSION_ID="${RAY_JOB_SUBMISSION_ID:-${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"

ray job submit --address="http://host_0:8265" \
   --submission-id "${RAY_JOB_SUBMISSION_ID}" \
   --no-wait \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u "${SLIME_DIR}/${TRAIN_ENTRY}" \
   "${MODE_ARGS[@]}" \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS}" \
   --num-gpus-per-node "${SLIME_GPUS_PER_NODE:-8}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISMATCH_ARGS[@]}" \
   "${WANDB_ARGS[@]}"

echo "Following live Ray logs for ${RAY_JOB_SUBMISSION_ID} (also: ${TRAIN_LOG})"
set +e
ray job logs --address="http://host_0:8265" "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record | tee -a "${TRAIN_LOG}"
RAY_LOG_EXIT=$?
RAY_FINAL_STATUS=$(python3 "${SCRIPT_DIR}/ray_job_status.py" "http://host_0:8265" "${RAY_JOB_SUBMISSION_ID}")
RAY_STATUS_EXIT=$?
echo "Ray final status: ${RAY_FINAL_STATUS}"
set -e

if (( RAY_STATUS_EXIT == 0 )) && [[ "${RAY_FINAL_STATUS}" == "SUCCEEDED" ]]; then
   exit 0
fi

echo "Ray job failed (submission id: ${RAY_JOB_SUBMISSION_ID}, logs exit: ${RAY_LOG_EXIT})"
exit 1
