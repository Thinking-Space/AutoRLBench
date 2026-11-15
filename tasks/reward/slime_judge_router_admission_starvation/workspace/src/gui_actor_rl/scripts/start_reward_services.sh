#!/usr/bin/env bash
# GUI Actor RL — reward services (frozen judge replicas + holo3 probe) on dedicated GPUs.
#
# Started from the *worker* rank of the training job, after Ray and the env workers,
# on the GPUs that are NOT handed to Ray. It brings up:
#
#   judge-0..N-1   sglang, tp=1 each, one GPU each   ─┐
#                                                     ├─ sglang_router → JUDGE_PORT
#   probe          holo3 35B-A3B, tp=PROBE_TP          (exposed on PROBE_PORT, used directly)
#
# On success it writes ${REWARD_LOG_DIR}/reward_services.env with
#
#   GUI_JUDGE_BASE_URL=http://host_0:${JUDGE_PORT}/v1
#   GUI_JUDGE_MODEL_NAME=...
#   GUI_PROBE_BASE_URL=http://host_0:${PROBE_PORT}/v1
#   GUI_PROBE_MODEL_NAME=...
#   GUI_PROBE_ENABLE=true|false
#
# Usage:
#   start_reward_services.sh start|stop|status|logs [worker-index|router|probe]
#
# Everything is env-overridable; see the config block below.

set -euo pipefail

ACTION="${1:-start}"
LOGSEL="${2:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REWARD_LOG_DIR="${REWARD_LOG_DIR:-/tmp/gui_reward_services}"
# Host written into reward_services.env. Set to the node IP when the trainer runs
# on a different rank than the services (host_0 only works same-pod).
REWARD_PUBLIC_HOST="${REWARD_PUBLIC_HOST:-host_0}"
mkdir -p "${REWARD_LOG_DIR}"

# ── judge ──────────────────────────────────────────────────────────
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${JUDGE_HF_CHECKPOINT:-/workspace/assets/judge-checkpoint/rollout_17}}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-guirl-judge}"
# ";"-separated GPU ids, one replica per entry (tp=1)
JUDGE_GPU_LIST="${JUDGE_GPU_LIST:-4;5}"
JUDGE_WORKER_PORT="${JUDGE_WORKER_PORT:-31101}"   # first worker; +i per replica
JUDGE_PORT="${JUDGE_PORT:-31100}"                 # router
JUDGE_MEM_FRACTION="${JUDGE_MEM_FRACTION:-0.60}"
JUDGE_READY_TIMEOUT="${JUDGE_READY_TIMEOUT:-1800}"
JUDGE_ROUTER_POLICY="${JUDGE_ROUTER_POLICY:-round_robin}"

# ── probe (holo3 35B-A3B) ──────────────────────────────────────────
PROBE_ENABLE="${PROBE_ENABLE:-1}"
PROBE_MODEL_PATH="${PROBE_MODEL_PATH:-/workspace/local/models/Holo-3.1-35B-A3B}"
PROBE_SERVED_NAME="${PROBE_SERVED_NAME:-holo3-1-35b-a3b}"
PROBE_GPUS="${PROBE_GPUS:-6,7}"                   # comma-separated, tp=PROBE_TP
PROBE_TP="${PROBE_TP:-2}"
PROBE_PORT="${PROBE_PORT:-31200}"
PROBE_BACKEND="${PROBE_BACKEND:-sglang}"          # sglang | vllm
PROBE_MEM_FRACTION="${PROBE_MEM_FRACTION:-0.65}"
# Mirrors external/pkgs/run_agentplat/scripts/start_holo31_35b_sglang.sh (the known-good
# Holo-3.1-35B-A3B recipe): qwen3 reasoning parser keeps <think> out of `content`, which
# the probe's JSON-schema parser depends on.
PROBE_REASONING_PARSER="${PROBE_REASONING_PARSER:-qwen3}"
PROBE_DTYPE="${PROBE_DTYPE:-bfloat16}"
PROBE_MAX_RUNNING_REQUESTS="${PROBE_MAX_RUNNING_REQUESTS:-64}"
PROBE_ATTENTION_BACKEND="${PROBE_ATTENTION_BACKEND:-flashinfer}"
PROBE_READY_TIMEOUT="${PROBE_READY_TIMEOUT:-3600}"
# 1 = abort if the probe cannot start, 0 = warn and continue with probe disabled
PROBE_REQUIRED="${PROBE_REQUIRED:-0}"

# Keep the allocator from failing on fragmentation under multimodal prefill spikes.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

IFS=';' read -r -a JUDGE_GPUS <<< "${JUDGE_GPU_LIST}"
NUM_JUDGE=${#JUDGE_GPUS[@]}

judge_pidfile() { echo "${REWARD_LOG_DIR}/judge-$1.pid"; }
judge_log()     { echo "${REWARD_LOG_DIR}/judge-$1.log"; }
judge_url()     { echo "http://host_0:$(( JUDGE_WORKER_PORT + $1 ))"; }
ROUTER_PIDFILE="${REWARD_LOG_DIR}/judge-router.pid"
ROUTER_LOG="${REWARD_LOG_DIR}/judge-router.log"
PROBE_PIDFILE="${REWARD_LOG_DIR}/probe.pid"
PROBE_LOG="${REWARD_LOG_DIR}/probe.log"
ENVFILE="${REWARD_LOG_DIR}/reward_services.env"

usage() {
  cat <<EOF
Usage: $(basename "$0") {start|supervise|stop|status|logs [N|router|probe]}

  start      bring the services up and return
  supervise  bring them up and stay in the foreground (the training launcher needs this)

Env overrides:
  JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH}
  JUDGE_GPU_LIST=${JUDGE_GPU_LIST}   (replicas=${NUM_JUDGE}, tp=1 each)
  JUDGE_PORT=${JUDGE_PORT} (router)  JUDGE_WORKER_PORT=${JUDGE_WORKER_PORT} (first worker)
  PROBE_ENABLE=${PROBE_ENABLE}  PROBE_MODEL_PATH=${PROBE_MODEL_PATH}
  PROBE_GPUS=${PROBE_GPUS}  PROBE_TP=${PROBE_TP}  PROBE_PORT=${PROBE_PORT}
  PROBE_BACKEND=${PROBE_BACKEND}  PROBE_REQUIRED=${PROBE_REQUIRED}
  REWARD_LOG_DIR=${REWARD_LOG_DIR}
EOF
}

wait_http() {  # url, pidfile, timeout_s, label, log
  local url="$1" pidfile="$2" timeout="$3" label="$4" log="$5"
  local deadline=$(( $(date +%s) + timeout ))
  printf '[%s] waiting for %s ' "${label}" "${url}"
  while true; do
    if curl -sf "${url}/health" >/dev/null 2>&1; then
      echo " ready"
      return 0
    fi
    if [[ -f "${pidfile}" ]] && ! kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
      echo ""
      echo "[${label}] ERROR: process died — see ${log}" >&2
      tail -40 "${log}" >&2 || true
      return 1
    fi
    if (( $(date +%s) > deadline )); then
      echo ""
      echo "[${label}] ERROR: not ready within ${timeout}s — see ${log}" >&2
      return 1
    fi
    printf '.'
    sleep 5
  done
}

start_judge_replica() {
  local i="$1" gpu="$2" port="$3"
  local pidfile; pidfile=$(judge_pidfile "$i")
  local log;     log=$(judge_log "$i")
  if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
    echo "[judge-$i] already running (pid $(cat "${pidfile}")) on GPU ${gpu}:${port}"
    return 0
  fi
  echo "[judge-$i] starting — GPU ${gpu}, port ${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m sglang.launch_server \
    --model-path "${JUDGE_MODEL_PATH}" \
    --served-model-name "${JUDGE_SERVED_NAME}" \
    --host host_1 \
    --port "${port}" \
    --tp-size 1 \
    --mem-fraction-static "${JUDGE_MEM_FRACTION}" \
    > "${log}" 2>&1 &
  echo $! > "${pidfile}"
}

start_probe() {
  if [[ "${PROBE_ENABLE}" != "1" ]]; then
    echo "[probe] disabled (PROBE_ENABLE=${PROBE_ENABLE})"
    return 0
  fi
  if [[ ! -d "${PROBE_MODEL_PATH}" ]]; then
    echo "[probe] ERROR: model path missing: ${PROBE_MODEL_PATH}" >&2
    [[ "${PROBE_REQUIRED}" == "1" ]] && return 1
    PROBE_ENABLE=0
    return 0
  fi
  if [[ -f "${PROBE_PIDFILE}" ]] && kill -0 "$(cat "${PROBE_PIDFILE}")" 2>/dev/null; then
    echo "[probe] already running (pid $(cat "${PROBE_PIDFILE}")) on GPUs ${PROBE_GPUS}"
    return 0
  fi
  echo "[probe] starting — GPUs ${PROBE_GPUS} (tp=${PROBE_TP}), port ${PROBE_PORT}, backend ${PROBE_BACKEND}"
  case "${PROBE_BACKEND}" in
    sglang)
      CUDA_VISIBLE_DEVICES="${PROBE_GPUS}" "${PYTHON_BIN}" -m sglang.launch_server \
        --model-path "${PROBE_MODEL_PATH}" \
        --served-model-name "${PROBE_SERVED_NAME}" \
        --host host_1 \
        --port "${PROBE_PORT}" \
        --tp-size "${PROBE_TP}" \
        --mem-fraction-static "${PROBE_MEM_FRACTION}" \
        --trust-remote-code \
        --dtype "${PROBE_DTYPE}" \
        --max-running-requests "${PROBE_MAX_RUNNING_REQUESTS}" \
        --attention-backend "${PROBE_ATTENTION_BACKEND}" \
        --reasoning-parser "${PROBE_REASONING_PARSER}" \
        > "${PROBE_LOG}" 2>&1 &
      ;;
    vllm)
      CUDA_VISIBLE_DEVICES="${PROBE_GPUS}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model "${PROBE_MODEL_PATH}" \
        --served-model-name "${PROBE_SERVED_NAME}" \
        --host host_1 \
        --port "${PROBE_PORT}" \
        --tensor-parallel-size "${PROBE_TP}" \
        --trust-remote-code \
        --dtype "${PROBE_DTYPE}" \
        --max-num-seqs "${PROBE_MAX_RUNNING_REQUESTS}" \
        --reasoning-parser holo2 \
        > "${PROBE_LOG}" 2>&1 &
      ;;
    *)
      echo "[probe] ERROR: unknown PROBE_BACKEND=${PROBE_BACKEND}" >&2
      return 1
      ;;
  esac
  echo $! > "${PROBE_PIDFILE}"
}

write_envfile() {
  local probe_enabled="false"
  if [[ "${PROBE_ENABLE}" == "1" ]] && [[ -f "${PROBE_PIDFILE}" ]] \
     && kill -0 "$(cat "${PROBE_PIDFILE}")" 2>/dev/null; then
    probe_enabled="true"
  fi
  cat > "${ENVFILE}" <<EOF
# generated by start_reward_services.sh at $(date -u +%Y-%m-%dT%H:%M:%SZ)
export GUI_REWARD_SOURCE=judge
export GUI_JUDGE_EXTERNAL=1
export GUI_JUDGE_BASE_URL=http://${REWARD_PUBLIC_HOST}:${JUDGE_PORT}/v1
export GUI_JUDGE_MODEL_NAME=${JUDGE_SERVED_NAME}
export GUI_PROBE_ENABLE=${probe_enabled}
export GUI_PROBE_BASE_URL=http://${REWARD_PUBLIC_HOST}:${PROBE_PORT}/v1
export GUI_PROBE_MODEL_NAME=${PROBE_SERVED_NAME}
EOF
  echo "wrote ${ENVFILE}"
  cat "${ENVFILE}"
}

do_start() {
  if [[ ! -d "${JUDGE_MODEL_PATH}" ]]; then
    echo "ERROR: JUDGE_MODEL_PATH not found: ${JUDGE_MODEL_PATH}" >&2
    exit 1
  fi
  echo "=== reward services (judge replicas=${NUM_JUDGE}, probe=${PROBE_ENABLE}) ==="
  echo "  judge model:  ${JUDGE_MODEL_PATH}"
  echo "  probe model:  ${PROBE_MODEL_PATH} (tp=${PROBE_TP}, GPUs ${PROBE_GPUS})"

  local i gpu port
  for (( i=0; i<NUM_JUDGE; i++ )); do
    gpu="${JUDGE_GPUS[$i]}"
    port=$(( JUDGE_WORKER_PORT + i ))
    start_judge_replica "${i}" "${gpu}" "${port}"
  done

  # probe starts in the background; judge readiness is the critical path
  start_probe || { echo "[probe] failed to start" >&2; exit 1; }

  for (( i=0; i<NUM_JUDGE; i++ )); do
    wait_http "$(judge_url "$i")" "$(judge_pidfile "$i")" "${JUDGE_READY_TIMEOUT}" "judge-$i" "$(judge_log "$i")"
  done

  if (( NUM_JUDGE > 1 )); then
    if [[ -f "${ROUTER_PIDFILE}" ]] && kill -0 "$(cat "${ROUTER_PIDFILE}")" 2>/dev/null; then
      echo "[router] already running (pid $(cat "${ROUTER_PIDFILE}"))"
    else
      local worker_urls=()
      for (( i=0; i<NUM_JUDGE; i++ )); do
        worker_urls+=("--worker-urls" "$(judge_url "$i")")
      done
      echo "[router] :${JUDGE_PORT} → ${NUM_JUDGE} workers (${JUDGE_ROUTER_POLICY})"
      "${PYTHON_BIN}" -m sglang_router.launch_router \
        "${worker_urls[@]}" \
        --host host_1 \
        --port "${JUDGE_PORT}" \
        --policy "${JUDGE_ROUTER_POLICY}" \
        > "${ROUTER_LOG}" 2>&1 &
      echo $! > "${ROUTER_PIDFILE}"
      wait_http "http://host_0:${JUDGE_PORT}" "${ROUTER_PIDFILE}" 300 "router" "${ROUTER_LOG}"
    fi
  else
    echo "[router] single judge replica — exposing :${JUDGE_PORT} directly is unnecessary"
    # keep the contract stable: point the env file at the single worker port
    JUDGE_PORT=$(( JUDGE_WORKER_PORT ))
  fi

  if [[ "${PROBE_ENABLE}" == "1" ]]; then
    if wait_http "http://host_0:${PROBE_PORT}" "${PROBE_PIDFILE}" "${PROBE_READY_TIMEOUT}" "probe" "${PROBE_LOG}"; then
      :
    else
      if [[ "${PROBE_REQUIRED}" == "1" ]]; then
        echo "[probe] required but not ready — aborting" >&2
        exit 1
      fi
      echo "[probe] WARNING: probe not ready; continuing with GUI_PROBE_ENABLE=false" >&2
      PROBE_ENABLE=0
    fi
  fi

  write_envfile
  echo ""
  echo "============================================================"
  echo " reward services ready"
  echo "   judge : http://${REWARD_PUBLIC_HOST}:${JUDGE_PORT}/v1  (${JUDGE_SERVED_NAME}, ${NUM_JUDGE} replica(s))"
  if [[ "${PROBE_ENABLE}" == "1" ]]; then
    echo "   probe : http://${REWARD_PUBLIC_HOST}:${PROBE_PORT}/v1  (${PROBE_SERVED_NAME})"
  else
    echo "   probe : DISABLED"
  fi
  echo "============================================================"
}

kill_pidfile() {
  local pidfile="$1" label="$2"
  if [[ -f "${pidfile}" ]]; then
    local pid; pid=$(cat "${pidfile}")
    if kill -0 "${pid}" 2>/dev/null; then
      echo "stopping ${label} (pid ${pid})"
      kill "${pid}" 2>/dev/null || true
      for _ in $(seq 1 20); do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${pidfile}"
  fi
}

do_stop() {
  local i
  for (( i=0; i<NUM_JUDGE; i++ )); do kill_pidfile "$(judge_pidfile "$i")" "judge-$i"; done
  kill_pidfile "${ROUTER_PIDFILE}" "judge-router"
  kill_pidfile "${PROBE_PIDFILE}" "probe"
  rm -f "${ENVFILE}"
  echo "all reward services stopped"
}

do_status() {
  local i port
  for (( i=0; i<NUM_JUDGE; i++ )); do
    port=$(( JUDGE_WORKER_PORT + i ))
    if [[ -f "$(judge_pidfile "$i")" ]] && kill -0 "$(cat "$(judge_pidfile "$i")")" 2>/dev/null; then
      local h; h=$(curl -sf "http://host_0:${port}/health" >/dev/null 2>&1 && echo healthy || echo starting)
      echo "  judge-$i [GPU ${JUDGE_GPUS[$i]}]: running — :${port} (${h})"
    else
      echo "  judge-$i: NOT running"
    fi
  done
  if [[ -f "${ROUTER_PIDFILE}" ]] && kill -0 "$(cat "${ROUTER_PIDFILE}")" 2>/dev/null; then
    echo "  router: running — :${JUDGE_PORT}"
  else
    echo "  router: NOT running"
  fi
  if [[ "${PROBE_ENABLE}" == "1" ]]; then
    if [[ -f "${PROBE_PIDFILE}" ]] && kill -0 "$(cat "${PROBE_PIDFILE}")" 2>/dev/null; then
      local h; h=$(curl -sf "http://host_0:${PROBE_PORT}/health" >/dev/null 2>&1 && echo healthy || echo starting)
      echo "  probe [GPUs ${PROBE_GPUS}]: running — :${PROBE_PORT} (${h})"
    else
      echo "  probe: NOT running"
    fi
  else
    echo "  probe: disabled"
  fi
}

# Stay in the foreground so the training launcher (which aborts on any non-trainer
# child exiting, even with rc=0) keeps us alive for the whole run.
do_supervise() {
  local rc=0
  do_start || rc=$?
  if (( rc != 0 )); then
    echo "[supervise] startup failed (rc=${rc}); not supervising" >&2
    exit "${rc}"
  fi
  trap 'echo "[supervise] stopping reward services"; do_stop; exit 0' TERM INT
  echo "[supervise] entered supervision loop"
  local dead_streak=0
  while true; do
    sleep 30
    local judge_down=0
    local i
    for (( i=0; i<NUM_JUDGE; i++ )); do
      if [[ -f "$(judge_pidfile "$i")" ]] && ! kill -0 "$(cat "$(judge_pidfile "$i")")" 2>/dev/null; then
        echo "[supervise] ERROR: judge-$i died" >&2
        judge_down=1
      fi
    done
    if (( NUM_JUDGE > 1 )) && [[ -f "${ROUTER_PIDFILE}" ]] \
       && ! kill -0 "$(cat "${ROUTER_PIDFILE}")" 2>/dev/null; then
      echo "[supervise] ERROR: judge router died" >&2
      judge_down=1
    fi
    if (( judge_down )); then
      dead_streak=$(( dead_streak + 1 ))
      if (( dead_streak >= 2 )); then
        echo "[supervise] judge tier is down; aborting the run" >&2
        do_stop
        exit 1
      fi
    else
      dead_streak=0
    fi
    if [[ "${PROBE_ENABLE}" == "1" ]] && [[ -f "${PROBE_PIDFILE}" ]] \
       && ! kill -0 "$(cat "${PROBE_PIDFILE}")" 2>/dev/null; then
      echo "[supervise] WARNING: probe died — uncertain verdicts will be dropped" >&2
      if [[ "${PROBE_REQUIRED}" == "1" ]]; then do_stop; exit 1; fi
    fi
  done
}

do_logs() {
  case "${LOGSEL}" in
    probe)  tail -n 200 -f "${PROBE_LOG}" ;;
    router) tail -n 200 -f "${ROUTER_LOG}" ;;
    ''|*[!0-9]*) echo "Usage: $(basename "$0") logs [0-$((NUM_JUDGE-1))|router|probe]" >&2; exit 1 ;;
    *)      tail -n 200 -f "$(judge_log "${LOGSEL}")" ;;
  esac
}

case "${ACTION}" in
  start)  do_start ;;
  stop)   do_stop ;;
  restart) do_stop; sleep 3; do_start ;;
  supervise) do_supervise ;;
  status) do_status ;;
  logs)   do_logs ;;
  *) usage; exit 1 ;;
esac
