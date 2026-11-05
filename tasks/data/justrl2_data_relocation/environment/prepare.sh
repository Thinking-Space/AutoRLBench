#!/usr/bin/env bash
# justrl2_data_relocation — prepare stage.
#
# Runs in the prepare-stage container: root, the justrl2-data-relocation
# image, the only networked stage, and $AUTORLBENCH_WORKSPACE and
# $AUTORLBENCH_PREPARED writable. The task's own asset_tools download and
# verify every public asset against the assets.json / sources.lock.json pins
# (sha256 + size); the task-local fetch_hf.py only warms the shared model
# cache so the 5 GB checkpoint is fetched once per machine, and the framework
# tarball is fetched by the task's own pinned fetch_asset. In the agent stage
# this file is mounted read-only at /autorlbench/environment.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}" "${AUTORLBENCH_PREPARED:?}" "${AUTORLBENCH_ENVIRONMENT:?}"
workspace="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"
cd "$workspace"
mkdir -p "$workspace/cache" "$workspace/public-source"

# --- Base model: shared-cache fetch ------------------------------------------
# openbmb/JustRL-II-base-model @ b558bbc71d9433e7c5869effcbac07d4c7f2c5b4,
# the revision assets.json pins. Whole-repo fetch, cache-first, no checksum
# verification here -- prepare_assets.py re-verifies every file it stages.
python3 "$env_dir/fetch_hf.py" \
    --repo openbmb/JustRL-II-base-model \
    --dest "$workspace/cache/model"

# --- Framework source tarball -------------------------------------------------
# Fetched by the task's own fetch_asset with the assets.json pin (the single
# source of truth for its sha256), into the asset cache prepare_assets.py
# stages from. merak0514/JustRL-II @ 75814318f243af7e39af337c1173fc3927c9635b.
python3 -B - "$env_dir" "$workspace/cache/source.tar.gz" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/asset_tools")
from prepare_assets import fetch_asset, read_json, HERE
fetch_asset(sys.argv[2], read_json(HERE / "assets.json")["public_source_archive"])
PY

# --- Prepared assets -----------------------------------------------------------
# Stages the model and framework source from the cache (verified against
# assets.json), converts the checkpoint to safetensors, and downloads the
# four pinned public Math shards (openbmb/UltraData-RL-2609 @ e6ecfa73,
# pinned in sources.lock.json) itself.
assets="$AUTORLBENCH_PREPARED/assets"
if [[ ! -f "$assets/prepared_receipt.json" ]]; then
  python3 -B "$env_dir/asset_tools/prepare_assets.py" \
    --out "$assets" \
    --asset-dir "$workspace/cache" \
    --download
fi
python3 -B "$env_dir/setup_incident.py"
