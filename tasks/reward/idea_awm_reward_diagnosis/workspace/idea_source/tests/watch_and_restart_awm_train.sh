#!/usr/bin/env bash
# Watch tmux pane zgc_1:0 for the AWM OPSD trainer and restart it if it dies
# before finishing.  Normal completion is not treated as a crash.
set -u

TARGET_SESSION="${TARGET_SESSION:-zgc_1}"
TARGET_WINDOW="${TARGET_WINDOW:-0}"
TARGET_PANE="${TARGET_SESSION}:${TARGET_WINDOW}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/idea_source}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-examples/idea_trainer/run_awm_opsd_qwen3_8b.sh}"
CONDA_ENV="${CONDA_ENV:-seed}"
POLL_SECONDS="${POLL_SECONDS:-30}"
RESTART_COOLDOWN="${RESTART_COOLDOWN:-90}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/awm_opsd_qwen3_8b_step693_wandb_v2}"
TOTAL_STEPS="${TOTAL_STEPS:-200}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/outputs/awm_watchdog}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/watch_zgc1_0.log}"

log() {
    local ts
    ts="$(date '+%F %T')"
    echo "[$ts] $*" | tee -a "$LOG_FILE"
}

target_exists() {
    tmux has-session -t "$TARGET_SESSION" 2>/dev/null \
        && tmux list-panes -t "$TARGET_PANE" >/dev/null 2>&1
}

pane_pid() {
    tmux display-message -t "$TARGET_PANE" -p '#{pane_pid}'
}

descendants() {
    local root="$1"
    local kids
    kids="$(pgrep -P "$root" 2>/dev/null || true)"
    local pid
    for pid in $kids; do
        echo "$pid"
        descendants "$pid"
    done
}

training_running() {
    local root
    root="$(pane_pid)"
    [[ -n "$root" && -d "/proc/$root" ]] || return 1

    local pids
    pids="$(descendants "$root")"
    [[ -n "$pids" ]] || return 1

    local pid cmd
    for pid in $pids; do
        cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
        [[ -n "$cmd" ]] || continue
        if [[ "$cmd" == *"$TRAIN_SCRIPT"* || "$cmd" == *"verl.trainer.main_ppo"* ]]; then
            return 0
        fi
    done
    return 1
}

pane_tail() {
    tmux capture-pane -t "$TARGET_PANE" -p -S -120 2>/dev/null || true
}

training_finished() {
    local text
    text="$(pane_tail)"
    if grep -Eq "Final validation metrics:" <<<"$text"; then
        return 0
    fi
    if grep -Eq "Training Progress:[[:space:]]*100%" <<<"$text"; then
        return 0
    fi
    if grep -Eq "${TOTAL_STEPS}/${TOTAL_STEPS}" <<<"$text" && grep -Eq "Training Progress:" <<<"$text"; then
        return 0
    fi

    local tracker="$OUTPUT_DIR/latest_checkpointed_iteration.txt"
    if [[ -f "$tracker" ]]; then
        local step
        step="$(tr -d '[:space:]' < "$tracker" || true)"
        if [[ "$step" =~ ^[0-9]+$ ]] && (( step >= TOTAL_STEPS )); then
            return 0
        fi
    fi
    if [[ -d "$OUTPUT_DIR/global_step_${TOTAL_STEPS}" ]]; then
        return 0
    fi
    return 1
}

pane_at_prompt() {
    local last
    last="$(tmux capture-pane -t "$TARGET_PANE" -p -S -8 2>/dev/null | sed '/^$/d' | tail -1 || true)"
    [[ "$last" == *"$ "* || "$last" == *"> "* ]]
}

cleanup_leftovers() {
    log "Cleaning leftover trainer / Ray / AWM MCP processes before restart"
    pkill -u "$(id -u)" -f "verl.trainer.main_ppo" 2>/dev/null || true
    pkill -u "$(id -u)" -f "$TRAIN_SCRIPT" 2>/dev/null || true
    if command -v ray >/dev/null 2>&1; then
        ray stop --force >/dev/null 2>&1 || true
    fi
    if [[ -f "$PROJECT_ROOT/tests/cleanup_awm_mcp.py" ]]; then
        python3 "$PROJECT_ROOT/tests/cleanup_awm_mcp.py" --kill \
            --work-root "$OUTPUT_DIR/awm_runtime" >/dev/null 2>&1 || true
    fi
    sleep 3
}

restart_training() {
    if training_running; then
        log "Skip restart: training is already running"
        return 0
    fi

    cleanup_leftovers

    if ! pane_at_prompt; then
        log "Pane is not at a shell prompt; sending Ctrl-C first"
        tmux send-keys -t "$TARGET_PANE" C-c
        sleep 2
        tmux send-keys -t "$TARGET_PANE" C-c
        sleep 1
    fi

    log "Restarting: bash $TRAIN_SCRIPT"
    tmux send-keys -t "$TARGET_PANE" "cd $(printf %q "$PROJECT_ROOT")" Enter
    tmux send-keys -t "$TARGET_PANE" "conda activate $(printf %q "$CONDA_ENV")" Enter
    tmux send-keys -t "$TARGET_PANE" "bash $(printf %q "$TRAIN_SCRIPT")" Enter

    local waited=0
    while (( waited < RESTART_COOLDOWN )); do
        sleep 5
        waited=$((waited + 5))
        if training_running; then
            log "Training process is up after ${waited}s"
            return 0
        fi
    done
    log "WARNING: training process not detected ${RESTART_COOLDOWN}s after restart"
    return 1
}

log "Watchdog started. target=$TARGET_PANE script=$TRAIN_SCRIPT poll=${POLL_SECONDS}s"
log "Log file: $LOG_FILE"

if ! target_exists; then
    log "ERROR: tmux target $TARGET_PANE does not exist"
    exit 1
fi

finished_notified=0
while true; do
    if ! target_exists; then
        log "ERROR: tmux target $TARGET_PANE disappeared; exiting"
        exit 1
    fi

    if training_running; then
        finished_notified=0
        log "OK: training is running in $TARGET_PANE"
    elif training_finished; then
        if (( finished_notified == 0 )); then
            log "Training appears complete; not restarting"
            finished_notified=1
        fi
    else
        log "DETECTED interrupt/crash in $TARGET_PANE; restarting"
        restart_training || true
    fi
    sleep "$POLL_SECONDS"
done
