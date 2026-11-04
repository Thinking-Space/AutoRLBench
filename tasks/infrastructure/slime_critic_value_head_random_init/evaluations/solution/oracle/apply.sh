#!/usr/bin/env bash
# Apply the oracle fix to a workspace.
#
#     ./apply.sh /path/to/buggy_workspace
#
# One hunk in slime/slime/backends/megatron_utils/model_provider.py:
# zero-initialise the critic's value head instead of drawing it from
# N(0, init_method_std). Idempotent -- running it on an already-fixed workspace
# reports that and exits 0.
set -euo pipefail

WS="${1:?usage: apply.sh <workspace-dir>}"
PATCH="$(cd "$(dirname "$0")" && pwd)/value_head_zero_init.patch"
TARGET="${WS}/slime/slime/backends/megatron_utils/model_provider.py"

[ -f "${TARGET}" ] || { echo "no such file: ${TARGET}" >&2; exit 1; }

if patch -p1 -d "${WS}" --fuzz=0 --dry-run --reverse --force < "${PATCH}" >/dev/null 2>&1; then
    echo "already fixed: ${TARGET}"
    exit 0
fi

patch -p1 -d "${WS}" --fuzz=0 --no-backup-if-mismatch < "${PATCH}"

# The fix is only meaningful if it holds at construction time, so assert it
# rather than trusting that the patch landed somewhere useful. The checker is
# the verifier's copy (this script's sibling under ../verifier/tests/) -- it
# no longer ships inside the workspace.
CHECKER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../verifier/tests/check_value_head.py"
python3 "${CHECKER}" "${TARGET}"
