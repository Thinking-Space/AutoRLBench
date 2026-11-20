#!/usr/bin/env bash
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_ENVIRONMENT:?}"

# Qwen/Qwen3.5-0.8B at 2fc06364715b967f1860aea9cf38778875588b17.
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
model="$AUTORLBENCH_WORKSPACE/assets/model"
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
    --allow '*.json' --allow '*.safetensors' --allow '*.jinja' --allow merges.txt --dest "$model"

python3 - "$model" "$AUTORLBENCH_ENVIRONMENT/source_manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, manifest = map(Path, sys.argv[1:])
assets = json.loads(manifest.read_text())["model"]["files"]
for name, expected in assets.items():
    file = root / name
    if file.stat().st_size != expected["bytes"]:
        raise ValueError(f"Public checkpoint size mismatch for {name}")
    with file.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected["sha256"]:
        raise ValueError(f"Public checkpoint checksum mismatch for {name}")
PY
