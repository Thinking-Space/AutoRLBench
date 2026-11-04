#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
# shellcheck source=../../_common/teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft/_common/teacher_naming.sh"

MODELS_ROOT="${MODELS_ROOT:?Please set MODELS_ROOT in $ENV_FILE.}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/webshop_episode_skill_pipeline}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/filtered_after_train_${RUN_ID}.log}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

TRAIN_TMUX_SESSION="${TRAIN_TMUX_SESSION:-train}"
WAIT_FOR_TRAIN_TMUX="${WAIT_FOR_TRAIN_TMUX:-1}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"
POST_TRAIN_GPU_QUIET_SECONDS="${POST_TRAIN_GPU_QUIET_SECONDS:-60}"
POST_TRAIN_GPU_WAIT_TIMEOUT="${POST_TRAIN_GPU_WAIT_TIMEOUT:-900}"

PIPELINE_OUTPUT_DIR="${PIPELINE_OUTPUT_DIR:-$PROJECT_ROOT/outputs/webshop_episode_skill_pipeline_filtered_qwen25_3b_${SFT_SELF_DIR_SUFFIX}}"
PIPELINE_OVERWRITE="${PIPELINE_OVERWRITE:-true}"
PIPELINE_NUM_TASKS="${PIPELINE_NUM_TASKS:-360}"
PIPELINE_ROLLOUTS_PER_TASK="${PIPELINE_ROLLOUTS_PER_TASK:-8}"
PIPELINE_TASK_BATCH_SIZE="${PIPELINE_TASK_BATCH_SIZE:-16}"
PIPELINE_BASELINE_HISTORY_LENGTH="${PIPELINE_BASELINE_HISTORY_LENGTH:-5}"
PIPELINE_SKILL_GEN_WORKERS="${PIPELINE_SKILL_GEN_WORKERS:-128}"
PIPELINE_SKILL_PARSE_ATTEMPTS="${PIPELINE_SKILL_PARSE_ATTEMPTS:-2}"
PIPELINE_REQUEST_WORKERS="${PIPELINE_REQUEST_WORKERS:-128}"
PIPELINE_SFT_MIN_SOURCE_SCORE="${PIPELINE_SFT_MIN_SOURCE_SCORE:-0.4}"
PIPELINE_SFT_INCLUDE_SUCCESS="${PIPELINE_SFT_INCLUDE_SUCCESS:-true}"
PIPELINE_SFT_MAX_ZERO_SCORE_FAILURES="${PIPELINE_SFT_MAX_ZERO_SCORE_FAILURES:-256}"
PIPELINE_SFT_MAX_RECORDS="${PIPELINE_SFT_MAX_RECORDS:-}"

SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-3}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-12288}"
SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-8}"
SFT_MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-1}"
SFT_BASE_MODEL_PATH="${SFT_BASE_MODEL_PATH:-$MODELS_ROOT/Qwen2.5-3B-Instruct}"
SFT_EXPORT_MODEL_DIR="${SFT_EXPORT_MODEL_DIR:-$MODELS_ROOT/Qwen2.5-3B-Instruct-webshop-episode-skill-sft-${SFT_SELF_SUFFIX}}"

RL_SCRIPT="${RL_SCRIPT:-$PROJECT_ROOT/examples/idea_trainer/run_webshop_sft_glm_self.sh}"
RL_EXPERIMENT_NAME="${RL_EXPERIMENT_NAME:-seed-grpo_qwen2.5_3b_webshop_episode_no_skill_loss_sft_filtered_policy-vllm}"
RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-$MODELS_ROOT/ckpt/$RL_EXPERIMENT_NAME}"

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

pane_has_foreground_child() {
    local pane_pid="$1"
    pgrep -P "$pane_pid" >/dev/null 2>&1
}

wait_for_train_tmux() {
    if [[ "$WAIT_FOR_TRAIN_TMUX" != "1" ]]; then
        echo "[$(timestamp)] WAIT_FOR_TRAIN_TMUX=$WAIT_FOR_TRAIN_TMUX, skip tmux wait."
        return
    fi

    echo "[$(timestamp)] Waiting for tmux session '$TRAIN_TMUX_SESSION' to finish its current foreground command."
    while true; do
        local pane_pid
        pane_pid="$(tmux display-message -p -t "$TRAIN_TMUX_SESSION:0" '#{pane_pid}' 2>/dev/null || true)"
        if [[ -z "$pane_pid" ]]; then
            echo "[$(timestamp)] tmux session '$TRAIN_TMUX_SESSION' is not present; continuing."
            return
        fi
        if ! pane_has_foreground_child "$pane_pid"; then
            echo "[$(timestamp)] tmux session '$TRAIN_TMUX_SESSION' is idle; continuing."
            return
        fi

        local last_line
        last_line="$(tmux capture-pane -pt "$TRAIN_TMUX_SESSION:0" -S -20 2>/dev/null | sed '/^[[:space:]]*$/d' | tail -n 1 || true)"
        echo "[$(timestamp)] '$TRAIN_TMUX_SESSION' still running. Last line: ${last_line:-<empty>}"
        sleep "$WAIT_INTERVAL_SECONDS"
    done
}

wait_for_post_train_gpu_quiet() {
    if [[ "$POST_TRAIN_GPU_QUIET_SECONDS" == "0" ]]; then
        return
    fi

    echo "[$(timestamp)] Waiting for post-train GPU/Ray cleanup."
    local deadline=$((SECONDS + POST_TRAIN_GPU_WAIT_TIMEOUT))
    local quiet_start=""
    while true; do
        local active
        active="$(
            ps -eo pid,ppid,stat,cmd \
                | grep -E 'verl\\.trainer\\.main_ppo|ray::|raylet|gcs_server|vllm serve' \
                | grep -v grep \
                | grep -v 'webshop_episode_skill_pipeline' \
                || true
        )"
        if [[ -z "$active" ]]; then
            if [[ -z "$quiet_start" ]]; then
                quiet_start="$SECONDS"
            fi
            if (( SECONDS - quiet_start >= POST_TRAIN_GPU_QUIET_SECONDS )); then
                echo "[$(timestamp)] Post-train cleanup quiet window reached."
                return
            fi
        else
            quiet_start=""
            if (( SECONDS >= deadline )); then
                echo "[$(timestamp)] Post-train cleanup wait timed out; continuing anyway."
                echo "$active" | head -20
                return
            fi
        fi
        sleep 10
    done
}

run_pipeline() {
    echo "[$(timestamp)] Rebuilding filtered WebShop SFT data."
    echo "  output dir:       $PIPELINE_OUTPUT_DIR"
    echo "  tasks:            $PIPELINE_NUM_TASKS"
    echo "  rollouts/task:    $PIPELINE_ROLLOUTS_PER_TASK"
    echo "  baseline history: $PIPELINE_BASELINE_HISTORY_LENGTH"
    echo "  skill parse tries:$PIPELINE_SKILL_PARSE_ATTEMPTS"
    echo "  sft min score:    $PIPELINE_SFT_MIN_SOURCE_SCORE"
    echo "  zero-score cap:   $PIPELINE_SFT_MAX_ZERO_SCORE_FAILURES"

    OUTPUT_DIR="$PIPELINE_OUTPUT_DIR" \
    OVERWRITE="$PIPELINE_OVERWRITE" \
    NUM_TASKS="$PIPELINE_NUM_TASKS" \
    ROLLOUTS_PER_TASK="$PIPELINE_ROLLOUTS_PER_TASK" \
    TASK_BATCH_SIZE="$PIPELINE_TASK_BATCH_SIZE" \
    BASELINE_HISTORY_LENGTH="$PIPELINE_BASELINE_HISTORY_LENGTH" \
    REQUEST_WORKERS="$PIPELINE_REQUEST_WORKERS" \
    SKILL_GEN_WORKERS="$PIPELINE_SKILL_GEN_WORKERS" \
    SKILL_PARSE_ATTEMPTS="$PIPELINE_SKILL_PARSE_ATTEMPTS" \
    SFT_MIN_SOURCE_SCORE="$PIPELINE_SFT_MIN_SOURCE_SCORE" \
    SFT_INCLUDE_SUCCESS="$PIPELINE_SFT_INCLUDE_SUCCESS" \
    SFT_MAX_ZERO_SCORE_FAILURES="$PIPELINE_SFT_MAX_ZERO_SCORE_FAILURES" \
    SFT_MAX_RECORDS="$PIPELINE_SFT_MAX_RECORDS" \
    bash "$PROJECT_ROOT/scripts/sft/webshop/prepare_data.sh"
}

run_sft() {
    echo "[$(timestamp)] Starting WebShop episode-skill SFT."
    DATA_DIR="$PIPELINE_OUTPUT_DIR" \
    MODEL_PATH="$SFT_BASE_MODEL_PATH" \
    TOTAL_EPOCHS="$SFT_TOTAL_EPOCHS" \
    MAX_LENGTH="$SFT_MAX_LENGTH" \
    TRAIN_BATCH_SIZE="$SFT_TRAIN_BATCH_SIZE" \
    MICRO_BATCH_SIZE_PER_GPU="$SFT_MICRO_BATCH_SIZE_PER_GPU" \
    EXPORT_MODEL_DIR="$SFT_EXPORT_MODEL_DIR" \
    bash "$PROJECT_ROOT/scripts/sft/webshop/train_sft.sh"
}

run_rl() {
    echo "[$(timestamp)] Starting WebShop RL from filtered SFT model."
    HF_MODEL_PATH="$SFT_EXPORT_MODEL_DIR" \
    MODEL_PATH="$SFT_EXPORT_MODEL_DIR" \
    EXPERIMENT_NAME="$RL_EXPERIMENT_NAME" \
    DEFAULT_LOCAL_DIR="$RL_OUTPUT_DIR" \
    bash "$RL_SCRIPT"
}

echo "[$(timestamp)] WebShop filtered after-train flow queued."
echo "  log file:       $LOG_FILE"
echo "  train tmux:     $TRAIN_TMUX_SESSION"
echo "  rl script:      $RL_SCRIPT"

wait_for_train_tmux
wait_for_post_train_gpu_quiet
run_pipeline
run_sft
run_rl
