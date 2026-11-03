#!/usr/bin/env bash
# Keep at most two checkpoints in the RAID checkpoint directory.  Whenever
# three or more global_step_* directories are present, move the smallest step
# to the trainer's regular output directory.
set -u

SOURCE_DIR="${SOURCE_DIR:-/local/user/models/opsd-tool/awm_opsd_qwen3_8b_step693_wandb_v3}"
DEST_DIR="${DEST_DIR:-/workspace/idea_source/outputs/awm_opsd_qwen3_8b_step693_wandb_v3}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LOG_FILE="${LOG_FILE:-$DEST_DIR/checkpoint_watcher.log}"

mkdir -p "$DEST_DIR"

log() {
    local ts
    ts="$(date '+%F %T')"
    echo "[$ts] $*" | tee -a "$LOG_FILE"
}

list_checkpoints() {
    find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d \
        -regextype posix-extended -regex '.*/global_step_[0-9]+' -printf '%f\n' \
        | sort -V
}

move_oldest_if_needed() {
    [[ -d "$SOURCE_DIR" ]] || { log "等待 checkpoint 目录: $SOURCE_DIR"; return; }

    local checkpoints count oldest destination
    checkpoints="$(list_checkpoints)"
    count="$(printf '%s\n' "$checkpoints" | sed '/^$/d' | wc -l)"
    if (( count < 2 )); then
        log "当前 checkpoint 数量: $count，继续监测"
        return
    fi

    oldest="$(printf '%s\n' "$checkpoints" | sed '/^$/d' | head -n 1)"
    [[ -n "$oldest" ]] || return
    destination="$DEST_DIR/$oldest"
    if [[ -e "$destination" ]]; then
        log "目标已存在，跳过移动: $destination"
        return
    fi

    log "checkpoint 数量为 $count，移动最小 step: $oldest -> $DEST_DIR"
    if mv -- "$SOURCE_DIR/$oldest" "$destination"; then
        log "移动完成: $destination"
    else
        log "ERROR: 移动失败: $SOURCE_DIR/$oldest"
    fi
}

log "Checkpoint watcher started (source=$SOURCE_DIR, destination=$DEST_DIR, poll=${POLL_SECONDS}s)"
while true; do
    move_oldest_if_needed
    sleep "$POLL_SECONDS"
done
