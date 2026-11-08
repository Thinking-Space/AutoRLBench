#!/usr/bin/env bash
# Qwen2.5 / HotpotQA — prepare stage.
# Runs in the network-enabled prepare container. Assets are cached, verified
# and materialized at the workspace paths consumed by run.py and serve.py.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
environment_dir="${AUTORLBENCH_ENVIRONMENT:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export PYTHONDONTWRITEBYTECODE=1

# --- Base model -------------------------------------------------------------
python3 "$environment_dir/fetch_hf.py" \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    --dest "$AUTORLBENCH_WORKSPACE/shared/models/actor"

# --- Retrieval encoder ------------------------------------------------------
python3 "$environment_dir/fetch_hf.py" \
    --repo intfloat/e5-base-v2 \
    --revision f52bf8ec8c7124536f0efb74aca902b2995e5bcd \
    --dest "$AUTORLBENCH_WORKSPACE/shared/models/encoder"

# --- Dataset ----------------------------------------------------------------
python3 "$environment_dir/fetch_hf.py" \
    --type dataset \
    --repo LIF1014/qwen25-hotpotqa-1024 \
    --revision d40bbbe3772947a360b6315f72392ab2e398e38a \
    --dest "$AUTORLBENCH_WORKSPACE/shared/data" \
    --manifest "$environment_dir/data_manifest.json"

# --- Derived indexes and workspace check ------------------------------------
python3 "$environment_dir/build_index.py"
python3 "$AUTORLBENCH_WORKSPACE/run.py" --check
