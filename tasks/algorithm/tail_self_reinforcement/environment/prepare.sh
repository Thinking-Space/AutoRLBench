#!/usr/bin/env bash
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then pip install numpy; fi
echo "environment ready"
