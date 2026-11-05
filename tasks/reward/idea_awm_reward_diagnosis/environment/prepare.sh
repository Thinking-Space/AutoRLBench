#!/usr/bin/env bash
# Fetch pinned dataset inputs during prepare, then verify all retained assets.
set -euo pipefail
env_dir="${AUTORLBENCH_ENVIRONMENT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
workspace="${AUTORLBENCH_WORKSPACE:-$(cd "$env_dir/../workspace" && pwd)}"
export PYTHONDONTWRITEBYTECODE=1

# --- image-level env, seeded here --------------------------------------------
# The recording rig exported PYTHONDONTWRITEBYTECODE=1; it rides the
# prepared/env handoff into the agent/rerun/verifier stages via the activate
# hook (WANDB_MODE=offline is exported by the harness itself for every
# stage). Idempotent.
if [[ -n "${AUTORLBENCH_PREPARED:-}" ]]; then
    if ! grep -qs "^PYTHONDONTWRITEBYTECODE=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "PYTHONDONTWRITEBYTECODE=1" >> "${AUTORLBENCH_PREPARED}/env"
    fi
fi

# The URL helper needs only Python's standard library; no extra HF SDK in the image.
while IFS=$'\t' read -r url sha256 size relative; do
    python3 "$env_dir/fetch_url.py" --type dataset --url "$url" \
        --sha256 "$sha256" --size "$size" --dest "$workspace/$relative"
done < <(python3 - "$env_dir/data_manifest.json" <<'PY'
import os
import json
import sys
from pathlib import PurePosixPath

manifest = json.load(open(sys.argv[1]))
source = manifest['download']
# Mirrored HF endpoint; huggingface.co itself is not on the cluster egress
# whitelist, so the mirror is the default and HF_ENDPOINT overrides it.
endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com').rstrip('/')
base = f"{endpoint}/datasets/{source['repo']}/resolve/{source['revision']}"
for entry in manifest['files']:
    relative = PurePosixPath(entry['path']).relative_to(source['destination'])
    print(f"{base}/{relative}\t{entry['sha256']}\t{entry['bytes']}\t{entry['path']}")
PY
)
exec python3 "$env_dir/check_assets.py" --workspace "$workspace" --manifest "$env_dir/data_manifest.json"
