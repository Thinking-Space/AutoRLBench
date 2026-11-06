#!/usr/bin/env bash
# Source only after selecting PYTHON and switching to the project root.
MAESTRO_LOG_DIR="$("$PYTHON" -m tools.log_paths "${1:-run}")" || return 1
export MAESTRO_LOG_DIR
[[ -n "$MAESTRO_LOG_DIR" && -d "$MAESTRO_LOG_DIR" ]] || return 1
