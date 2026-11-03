#!/usr/bin/env bash
# alfworld_tool_schema_mismatch — prepare stage.
#
# Runs in a separate prepare-stage container: root, `verl` image, **the only
# stage with network**. In the agent stage this file is mounted read-only at
# /autorlbench/environment.
#
# Recorded assets are fetched whole-repo with the task-local fetch_hf.py
# (cache-first under $AUTORLBENCH_MODELS, no revision/sha pins — converting to
# pinned autorlbench_fetch_hf would change fetch behavior).
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen3-4B \
    --dest "$AUTORLBENCH_WORKSPACE/tool_schema/Qwen3-4B"
