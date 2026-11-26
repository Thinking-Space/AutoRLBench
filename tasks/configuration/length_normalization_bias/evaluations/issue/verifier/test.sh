#!/usr/bin/env bash
set -euo pipefail
test -f "${1}/diagnosis.md" || exit 1
echo ok
