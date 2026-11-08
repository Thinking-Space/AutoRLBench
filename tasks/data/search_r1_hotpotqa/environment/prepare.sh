#!/usr/bin/env bash
# Prepare the pinned assets for the Qwen2.5 / HotpotQA configuration.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
environment_dir="${AUTORLBENCH_ENVIRONMENT:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export PYTHONDONTWRITEBYTECODE=1
python3 "$environment_dir/prepare_data.py"
python3 "$environment_dir/prepare_assets.py"
python3 "$environment_dir/build_index.py"
python3 "$AUTORLBENCH_WORKSPACE/run.py" --check
