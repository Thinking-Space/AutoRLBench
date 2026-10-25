#!/usr/bin/env bash
# Fetch pinned dataset inputs during prepare, then verify all retained assets.
set -euo pipefail
env_dir="${AUTORLBENCH_ENVIRONMENT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
workspace="${AUTORLBENCH_WORKSPACE:-$(cd "$env_dir/../workspace" && pwd)}"
export PYTHONDONTWRITEBYTECODE=1
# The URL helper needs only Python's standard library; no extra HF SDK in the image.
while IFS=$'\t' read -r url sha256 size relative; do
    autorlbench_fetch_url --type dataset --url "$url" \
        --sha256 "$sha256" --size "$size" --dest "$workspace/$relative"
done < <(python3 - "$env_dir/data_manifest.json" <<'PY'
import json
import sys
from pathlib import PurePosixPath

manifest = json.load(open(sys.argv[1]))
source = manifest['download']
base = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}"
for entry in manifest['files']:
    relative = PurePosixPath(entry['path']).relative_to(source['destination'])
    print(f"{base}/{relative}\t{entry['sha256']}\t{entry['bytes']}\t{entry['path']}")
PY
)
exec python3 "$env_dir/check_assets.py" --workspace "$workspace" --manifest "$env_dir/data_manifest.json"
