#!/usr/bin/env bash
set -euo pipefail
W="$AUTORLBENCH_WORKSPACE"
export TMPDIR="${TMPDIR:-$W/runtime/tmp}"
export XDG_CACHE_HOME="$W/runtime/cache"
export HF_HOME="$W/runtime/hf"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
# asset_pins.json records immutable model/data revisions and file hashes.
# The shared helper remains identical to the other tasks' helper.
python3 - "$AUTORLBENCH_ENVIRONMENT/asset_pins.json" <<'PY_PREPARE_ASSETS'
import hashlib
import json
import os
from pathlib import Path
import sys

pins = Path(sys.argv[1])
sys.path.insert(0, str(pins.parent))
from fetch_hf import _force_disable_xet, materialize, root_for
from huggingface_hub import snapshot_download

_force_disable_xet()
workspace = Path(os.environ['AUTORLBENCH_WORKSPACE'])
for asset in json.loads(pins.read_text())['assets']:
    snapshot = Path(snapshot_download(
        repo_id=asset['repo'], repo_type=asset['type'], revision=asset['revision'],
        cache_dir=root_for(asset['type']) / asset['cache_name'],
        allow_patterns=[entry['path'] for entry in asset['files']], max_workers=4,
    ))
    for entry in asset['files']:
        path = snapshot / entry['path']
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(16 * 1024**2), b''):
                digest.update(block)
        if path.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            raise ValueError('Asset differs from recorded bytes: ' + entry['path'])
    materialize(snapshot, workspace / asset['destination'])
PY_PREPARE_ASSETS
