#!/usr/bin/env bash
# prepare.sh —— 素材阶段只需 numpy。
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
    pip install numpy
fi
echo "environment ready"