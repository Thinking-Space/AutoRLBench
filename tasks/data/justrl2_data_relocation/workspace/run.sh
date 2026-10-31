#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
LOG_DIR=${LOG_DIR:-"$PWD/logs"}
mkdir -p "$LOG_DIR"
exec python3 run_case.py --arm faulty --steps 8 --relocate-at 4   --batch 8 --samples 2 --max-response 32768 --max-total-tokens 262144   --prepared ./prepared --output "$LOG_DIR/run-home"
