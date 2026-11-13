#!/usr/bin/env bash
set -euo pipefail
test -n "$AUTORLBENCH_WORKSPACE"
# This issue-only task ships a frozen code/log snapshot and has no model or dataset fetch.
