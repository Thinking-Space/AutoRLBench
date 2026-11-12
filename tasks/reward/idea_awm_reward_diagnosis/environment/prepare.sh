#!/usr/bin/env bash
# Fetch pinned dataset inputs during prepare, then verify all retained assets.
set -euo pipefail
env_dir="${AUTORLBENCH_ENVIRONMENT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
workspace="${AUTORLBENCH_WORKSPACE:-$(cd "$env_dir/../workspace" && pwd)}"
export PYTHONDONTWRITEBYTECODE=1

# --- image-level env, seeded here --------------------------------------------
# The recording rig exported PYTHONDONTWRITEBYTECODE=1 and WANDB_SILENT=true;
# they ride the prepared/env handoff into the agent/rerun/verifier stages via
# the activate hook (WANDB_MODE=offline is exported by the harness itself for
# every stage). Idempotent.
for _kv in PYTHONDONTWRITEBYTECODE=1 WANDB_SILENT=true; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# --- recorded diagnosis pins, shadowed over the image -------------------------
# The task runs on the packaged vllm0.11-torch2.8-cu128 image (py3.12; the
# recorded idea-issue environment was CPU-only python:3.12-slim). The scoring
# surface is environment-independent -- the issue judge runs outside the
# sandbox and the solution grader is stdlib-only -- so what the image must
# supply is exactly the evidence-reading layer: pyarrow (read-only parquet
# inspection), omegaconf (config/.hydra/config.yaml) and wandb (self-test
# imports; the frozen evidence itself is plain JSONL/text). Those and their
# transitive pins -- the 23-entry recorded freeze in
# runtime/diagnosis-requirements.txt, byte-identical to the former
# docker-build/idea-issue.Dockerfile -- are shadowed from
# $AUTORLBENCH_PREPARED/pysite. --no-deps: the pins are the complete closure
# of the recorded diagnosis environment, and letting pip resolve would float
# pins against the image's own stack. One exception to wheel-pinning:
# antlr4-python3-runtime 4.9.3 (omegaconf's pinned parser) ships no wheel on
# PyPI -- sdist only -- so it installs from source with --no-build-isolation
# (the image's own setuptools builds the pure-Python setup.py).
site="$AUTORLBENCH_PREPARED/pysite"
if [ ! -d "$site/pyarrow" ]; then
    _pip_args=(--no-cache-dir --no-deps --target "$site")
    if [ -n "${AUTORLBENCH_PIP_INDEX_URL:-}" ]; then
        _pip_args+=(--index-url "$AUTORLBENCH_PIP_INDEX_URL")
        if [ -n "${AUTORLBENCH_PIP_TRUSTED_HOST:-}" ]; then
            _pip_args+=(--trusted-host "$AUTORLBENCH_PIP_TRUSTED_HOST")
        fi
    fi
    _wheels_req="$(mktemp)"
    grep -v -E '^(antlr4-python3-runtime)==' "$env_dir/runtime/diagnosis-requirements.txt" > "$_wheels_req"
    python3 -m pip install "${_pip_args[@]}" --only-binary :all: -r "$_wheels_req"
    python3 -m pip install "${_pip_args[@]}" --no-build-isolation \
        --no-deps 'antlr4-python3-runtime==4.9.3'
    rm -f "$_wheels_req"
fi
# Shadow for the rest of this script and for the agent/rerun/verifier stages
# via prepared/env. The explicit /workspace keeps the image's PYTHONPATH ENV
# in play -- the prepared/env line replaces it.
export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
if ! grep -qs "^PYTHONPATH=" "${AUTORLBENCH_PREPARED}/env"; then
    echo "PYTHONPATH=$site:/workspace" >> "${AUTORLBENCH_PREPARED}/env"
fi
PYTHONPATH="$site" python3 - <<'EOF'
import pyarrow, pyarrow.parquet, omegaconf, wandb
import importlib.metadata as m
assert pyarrow.__version__ == "25.0.1", pyarrow.__version__
assert omegaconf.__version__ == "2.3.1", omegaconf.__version__
assert m.version("wandb") == "0.28.2", m.version("wandb")
assert m.version("pydantic") == "2.13.4", m.version("pydantic")
assert m.version("protobuf") == "7.36.0", m.version("protobuf")
assert m.version("antlr4-python3-runtime") == "4.9.3", m.version("antlr4-python3-runtime")
EOF

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
