#!/usr/bin/env bash
# Source from the project root. Never log the contents of the private key file.
_wandb_restore_xtrace=0
case "$-" in
    *x*) _wandb_restore_xtrace=1; set +x ;;
esac
if [[ -f .secrets/wandb.env ]]; then
    source .secrets/wandb.env
fi

# Force local recording even if the caller previously selected online mode.
# Do not invoke `wandb login` or `wandb sync` from this training workflow.
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR:-${LOG_DIR:-.}}"
export WANDB_DATA_DIR=.cache/wandb/data
export WANDB_CACHE_DIR=.cache/wandb/cache
export WANDB_CONFIG_DIR=.cache/wandb/config
export WANDB_ERROR_REPORTING=false
export WANDB_DISABLE_GIT=true
export WANDB_DISABLE_CODE=true
export WANDB_SAVE_CODE=false
mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

if [[ "$_wandb_restore_xtrace" == 1 ]]; then
    unset _wandb_restore_xtrace
    set -x
else
    unset _wandb_restore_xtrace
fi
