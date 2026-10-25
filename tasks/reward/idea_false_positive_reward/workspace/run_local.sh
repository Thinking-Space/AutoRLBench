#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
train_python="${TRAIN_PYTHON:-/opt/idea/bin/python}"
if [[ ! -x "$train_python" ]]; then
    echo "Use the idea-historical shared image, or set TRAIN_PYTHON to an explicitly provisioned interpreter. No host conda environment is used." >&2
    exit 1
fi
# Authentication must come from HF_TOKEN, HF_TOKEN_PATH, or the user's HF login.
exec "$train_python" "$WORKSPACE/reproduce.py" "$@"
