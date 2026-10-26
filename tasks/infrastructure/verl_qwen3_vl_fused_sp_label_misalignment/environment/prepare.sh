#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONDONTWRITEBYTECODE=1
exec /opt/recorded-runtime/bin/python "$HERE/fetch_assets.py"
