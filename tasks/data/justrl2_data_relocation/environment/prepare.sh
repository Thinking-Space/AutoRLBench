#!/usr/bin/env bash
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}" "${AUTORLBENCH_PREPARED:?}" "${AUTORLBENCH_ENVIRONMENT:?}"
exec python3 -I -S "$AUTORLBENCH_ENVIRONMENT/setup_incident.py"
