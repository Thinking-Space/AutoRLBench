#!/usr/bin/env sh
# Installed as /etc/profile.d/autorlbench-activate.sh by every
# docker/images/*.Dockerfile.
#
# Every command the harness runs goes through `sh -lc`, so this file is the one
# place that shapes the per-stage environment. POSIX sh, not bash: it is sourced
# by /bin/sh and has no reason to need more.
#
# Each environment is its own upstream image, used as-is: the interpreter, the
# CUDA toolkit and the package set are whatever that image ships, so there is
# nothing to activate. What remains is three small things:
#
#   0. the image's PATH, if the login shell dropped part of it
#   1. what prepare.sh left behind            $AUTORLBENCH_PREPARED/env
#   2. where a stray `pip install` may write  agent stage only
#
# The task's workspace is not this file's business: the harness bind-mounts a
# per-run copy of it at /workspace before the container starts.

# --- 0. the image's PATH -----------------------------------------------------
# A login shell runs /etc/profile first, and on Debian that resets PATH to the
# stock list -- dropping whatever the image's ENV PATH added (/usr/local/cuda/bin,
# /root/.cargo/bin, ...). The Dockerfile recorded the image's PATH at build
# time; any entry missing now goes back in front, in the image's order. Where
# /etc/profile leaves PATH alone this changes nothing, and it is idempotent, so
# it runs outside the once-only guard below (a debugging `docker run ... sh -l`
# wants it too).
if [ -r /etc/autorlbench-image-path ]; then
    _autorlbench_image_path=""
    IFS= read -r _autorlbench_image_path < /etc/autorlbench-image-path || :
    _autorlbench_missing=""
    _autorlbench_ifs=$IFS
    IFS=:
    for _autorlbench_dir in $_autorlbench_image_path; do
        [ -n "$_autorlbench_dir" ] || continue
        case ":$_autorlbench_missing:$PATH:" in
            *":$_autorlbench_dir:"*) ;;
            *) _autorlbench_missing="${_autorlbench_missing:+$_autorlbench_missing:}$_autorlbench_dir" ;;
        esac
    done
    IFS=$_autorlbench_ifs
    if [ -n "$_autorlbench_missing" ]; then
        PATH="$_autorlbench_missing${PATH:+:$PATH}"
        export PATH
    fi
    unset _autorlbench_image_path _autorlbench_missing _autorlbench_ifs _autorlbench_dir
fi

# Nothing more to do outside a AutoRLBench sandbox (a plain `docker run` for
# debugging, say), and nothing to do twice.
if [ -z "${AUTORLBENCH_STAGE:-}" ] || [ -n "${AUTORLBENCH_ACTIVATED:-}" ]; then
    return 0 2>/dev/null || :
fi
AUTORLBENCH_ACTIVATED=1
export AUTORLBENCH_ACTIVATED

AUTORLBENCH_PREPARED="${AUTORLBENCH_PREPARED:-/autorlbench/prepared}"
AUTORLBENCH_MODELS="${AUTORLBENCH_MODELS:-/autorlbench/models}"
AUTORLBENCH_DATASETS="${AUTORLBENCH_DATASETS:-/autorlbench/datasets}"
export AUTORLBENCH_PREPARED AUTORLBENCH_MODELS AUTORLBENCH_DATASETS

# --- 1. what prepare.sh left behind ------------------------------------------
# The prepare stage records the paths it fetched to, plus whatever else the task
# needs exported, as KEY=VALUE lines. Plain assignments, not shell code: this
# file is sourced by every command in the agent and verifier stages, and a task
# package should not be able to run arbitrary shell in all of them.
#
# It lives under $AUTORLBENCH_PREPARED rather than in the workspace because the
# agent can write to the workspace, and the verifier reads this file too --
# grading must not consult anything the agent could have edited.
if [ -f "$AUTORLBENCH_PREPARED/env" ]; then
    while IFS= read -r _autorlbench_line || [ -n "$_autorlbench_line" ]; do
        case "$_autorlbench_line" in
            ''|'#'*) continue ;;
        esac
        _autorlbench_key="${_autorlbench_line%%=*}"
        # Reject anything that is not a bare NAME=..., so a malformed line
        # cannot turn into a command substitution when `export` parses it.
        case "$_autorlbench_key" in
            ''|*[!A-Za-z0-9_]*)
                echo "autorlbench: ignoring malformed line in prepared/env: $_autorlbench_line" >&2
                continue ;;
        esac
        export "$_autorlbench_line"
    done < "$AUTORLBENCH_PREPARED/env"
    unset _autorlbench_line _autorlbench_key
fi

# --- 2. where a stray install may write, agent and rerun stages --------------
# An agent that runs `pip install` should land in its own workspace rather than
# modify the image's site-packages -- and the verifier must not see it, because
# grading cannot depend on anything the agent installed. The rerun stage gets
# the same treatment: it reuses the agent's workspace, so anything the agent
# installed into PYTHONUSERBASE must be importable when the experiment runs, or
# a workspace that legitimately fixed a missing dependency would break at rerun
# time.
#
# PIP_BREAK_SYSTEM_PACKAGES is there for the Ubuntu 24.04 images, whose system
# interpreter is marked externally managed: pip refuses even a --user install
# there without it. With PIP_USER set the install still lands in the workspace
# user base, which is the thing that marker exists to protect.
if [ "$AUTORLBENCH_STAGE" = "agent" ] || [ "$AUTORLBENCH_STAGE" = "rerun" ]; then
    PYTHONUSERBASE="${PYTHONUSERBASE:-/workspace/.autorlbench-user}"
    PIP_USER=1
    PIP_BREAK_SYSTEM_PACKAGES=1
    export PYTHONUSERBASE PIP_USER PIP_BREAK_SYSTEM_PACKAGES
fi
