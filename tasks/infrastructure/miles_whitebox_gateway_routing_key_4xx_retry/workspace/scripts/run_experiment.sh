#!/bin/bash
# The task's experiment: real GRPO training with every rollout request driven
# through the white-box gateway chain, bounded by EXPERIMENT_WINDOW_S.
#
#   rollout plugin (public post(), 60-retry budget)   <- outer retry layer
#     -> gateway POST /sessions/{sid}/generate        <- the boundary under test
#        -> public post() (60-retry budget)           <- the retry helper
#           -> routing proxy (requires X-SMG-Routing-Key)
#              -> sglang router -> engine /generate
#
# Writes train/gateway/proxy logs and a machine-readable summary.json under
# $AUTORLBENCH_RESULTS/experiment_logs (defaulting to ./experiment_logs when
# the harness variables are absent). Exits 0 in every arm: grading reads the
# fresh run's logs, not this script's status.
set -uo pipefail

W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${AUTORLBENCH_RESULTS:-$W}"
LOGS="$RESULTS/experiment_logs"
WINDOW="${EXPERIMENT_WINDOW_S:-1200}"
NUM_ROLLOUT="${EXPERIMENT_NUM_ROLLOUT:-10}"

mkdir -p "$LOGS"

# --- environment --------------------------------------------------------------
# The image's training stack (ray / megatron / sglang) lives in /opt/sglang.
export PATH=/opt/sglang/bin:$PATH
# The workspace's miles tree shadows the image's editable install, and the
# workspace root carries the rollout plugin and the recipe.
export PYTHONPATH="$W/miles:$W${PYTHONPATH:+:$PYTHONPATH}"
# The pod's egress proxy refuses pod-IP destinations, and engines, gates and
# the routing proxy are all addressed by pod IP. Append, never replace.
POD_IP="$(hostname -I | awk '{print $1}')"
export no_proxy="127.0.0.1,localhost,${POD_IP},${no_proxy:-}"
export NO_PROXY="$no_proxy"

# --- clean slate --------------------------------------------------------------
/opt/sglang/bin/ray stop --force > /dev/null 2>&1 || true
pkill -f "strict_route[r].py" 2>/dev/null || true
pkill -f "whitebox_gatewa[y].py" 2>/dev/null || true
sleep 2

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/launch.log"; }

# --- start training -----------------------------------------------------------
cd "$W/miles"
setsid nohup /opt/sglang/bin/python "$W/recipes/run_training.py" \
      --num-rollout "$NUM_ROLLOUT" \
      > "$LOGS/train.log" 2>&1 < /dev/null &
TRAIN_PID=$!
log "training started (pid $TRAIN_PID, log: train.log)"

# --- wait for the router address, then start the routing proxy + gateway ------
ROUTER_ADDR=""
DEADLINE=$((SECONDS + 1800))
while [ $SECONDS -lt $DEADLINE ]; do
  if grep -q "Router ready at" "$LOGS/train.log" 2>/dev/null; then
    line="$(grep "Router ready at" "$LOGS/train.log" | head -1)"
    host="$(echo "$line" | grep -oE "host='[^']+'" | head -1 | sed "s/host='//;s/'//")"
    port="$(echo "$line" | grep -oE "port=[0-9]+" | head -1 | sed 's/port=//')"
    if [ -n "$host" ] && [ -n "$port" ]; then
      ROUTER_ADDR="$host:$port"
      break
    fi
  fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null && ! pgrep -f "run_trainin[g].py" > /dev/null; then
    log "training process died before the router became ready"
    break
  fi
  sleep 5
done

if [ -n "$ROUTER_ADDR" ]; then
  log "router at $ROUTER_ADDR"
  setsid nohup env ROUTER_SHIM_UPSTREAM="http://$ROUTER_ADDR" ROUTER_SHIM_PORT=30110 \
    /opt/sglang/bin/python "$W/routing_proxy/strict_router.py" > "$LOGS/proxy.log" 2>&1 < /dev/null &
  sleep 2
  setsid nohup env GATEWAY_ROUTING_PROXY_URL="http://127.0.0.1:30110" GATEWAY_PORT=30100 \
    /opt/sglang/bin/python "$W/gateway/whitebox_gateway.py" > "$LOGS/gateway.log" 2>&1 < /dev/null &
  log "routing proxy (30110 -> $ROUTER_ADDR) and gateway (30100 -> 30110) started"
else
  log "no router address discovered; training may have failed at startup"
fi

# --- monitor until the training exits or the window elapses --------------------
START=$SECONDS
while kill -0 "$TRAIN_PID" 2>/dev/null || pgrep -f "run_trainin[g].py" > /dev/null; do
  if [ $((SECONDS - START)) -ge "$WINDOW" ]; then
    log "experiment window (${WINDOW}s) elapsed; stopping the run"
    break
  fi
  sleep 10
done
ELAPSED=$((SECONDS - START))

# --- teardown ------------------------------------------------------------------
TRAINING_ALIVE=0
if pgrep -f "run_trainin[g].py" > /dev/null; then
  TRAINING_ALIVE=1
  pkill -f "run_trainin[g].py" 2>/dev/null || true
  sleep 5
fi
pkill -f "gateway_task_ru[n]" 2>/dev/null || true
pkill -f "strict_route[r].py" 2>/dev/null || true
pkill -f "whitebox_gatewa[y].py" 2>/dev/null || true
/opt/sglang/bin/ray stop --force > /dev/null 2>&1 || true
log "teardown complete (alive_at_window_end=$TRAINING_ALIVE, elapsed=${ELAPSED}s)"

# --- summary -------------------------------------------------------------------
count() { if [ -f "$2" ]; then grep -c "$1" "$2" || true; else echo 0; fi }

cat > "$LOGS/summary.json" <<EOF
{
  "window_s": $WINDOW,
  "elapsed_s": $ELAPSED,
  "num_rollout_budget": $NUM_ROLLOUT,
  "training_alive_at_window_end": $TRAINING_ALIVE,
  "rollout_iterations_completed": $(count "Final collected" "$LOGS/train.log"),
  "outer_retry_lines": $(count "retrying" "$LOGS/train.log"),
  "gateway_inner_retry_lines": $(count "retrying" "$LOGS/gateway.log"),
  "proxy_rejections": $(count "400 Bad" "$LOGS/proxy.log"),
  "gateway_server_errors": $(count " 500 Internal" "$LOGS/gateway.log"),
  "router_address": "${ROUTER_ADDR:-none}"
}
EOF
log "summary written to experiment_logs/summary.json"
exit 0
