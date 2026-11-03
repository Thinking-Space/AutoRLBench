#!/usr/bin/env bash
set -euo pipefail

# Run exactly one trainer iteration.  The underlying launcher performs data
# loading, student rollout, skill extraction, and OPD-loss computation inside
# that iteration, then exits without entering a second step.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/run_awm_opsd_qwen3_8b.sh"

[[ -f "$LAUNCHER" ]] || {
    echo "Launcher not found: $LAUNCHER" >&2
    exit 1
}

# Keep every setting overridable by the caller while choosing a lightweight,
# single-device default suitable for a one-iteration smoke run.
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export GROUP_SIZE="${GROUP_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
export TOTAL_TRAINING_STEPS=1

# Saving/testing is not needed for this probe; retain console output by
# default so the computed OPD loss is visible in the trainer logs.
export SAVE_FREQ="${SAVE_FREQ:-1000000}"
export TEST_FREQ="${TEST_FREQ:-1000000}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-['console']}"

# The base launcher defaults to resume_mode=auto.  Disable it explicitly so
# an existing OUTPUT_DIR can never cause this probe to load an old run.  A
# zero learning rate preserves the full backward/OPD-loss computation while
# preventing any model parameter update.
LOG_FILE="$(mktemp "${TMPDIR:-/tmp}/awm-opsd-one-step.XXXXXX.log")"
trap 'rm -f "$LOG_FILE"' EXIT

set +e
"$LAUNCHER" \
    trainer.resume_mode=disable \
    actor_rollout_ref.actor.optim.lr=0 \
    +actor_rollout_ref.actor.skip_optimizer_update=true \
    "$@" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

OPD_LOSS="$(grep -Eo "actor/opd_loss[=:][[:space:]]*[-+0-9.eE]+" "$LOG_FILE" | tail -1 | sed -E 's/.*[=:][[:space:]]*//')"
if [[ -n "$OPD_LOSS" ]]; then
    echo "OPD loss: $OPD_LOSS"
else
    echo "OPD loss was not found in trainer output" >&2
fi
exit "$STATUS"
