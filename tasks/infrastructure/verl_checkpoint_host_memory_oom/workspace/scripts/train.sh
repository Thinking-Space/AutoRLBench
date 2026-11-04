#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace PYTHONDONTWRITEBYTECODE=1
exec python3 launch.py "$@"
