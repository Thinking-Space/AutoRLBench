#!/usr/bin/env bash
# Prepare stage: the only networked stage. It fetches the pinned public policy
# model, regenerates the public workload, and installs the wheels the image
# lacks. Destinations are what workspace/scripts/run_config.json reads.
set -euo pipefail
ENVIRONMENT_DIR="${AUTORLBENCH_ENVIRONMENT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
W="${AUTORLBENCH_WORKSPACE:?prepare requires the staged workspace}"
PREPARED_ROOT="${AUTORLBENCH_PREPARED:?prepare requires the harness prepared directory}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

# --- Python wheels (exact bytes pinned in dependencies.json) -----------------
python3 -B "$ENVIRONMENT_DIR/prepare_dependencies.py" "$PREPARED_ROOT/dependencies"
if ! grep -qs '^PYTHONPATH=' "$PREPARED_ROOT/env" 2>/dev/null; then
    printf 'PYTHONPATH=%s\n' "$PREPARED_ROOT/dependencies/python" >> "$PREPARED_ROOT/env"
fi

# --- Policy model ------------------------------------------------------------
# Repo, 40-character revision and per-file SHA-256 digests live in
# model_manifest.json; fetch_hf.py refuses a tree that does not match them.
REPO=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["repo"])' "$ENVIRONMENT_DIR/model_manifest.json")
REVISION=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["revision"])' "$ENVIRONMENT_DIR/model_manifest.json")
python3 -B "$ENVIRONMENT_DIR/fetch_hf.py" --repo "$REPO" --revision "$REVISION" \
    --manifest "$ENVIRONMENT_DIR/model_manifest.json" --dest "$W/models/Qwen3.5-9B"

# --- Public workload -----------------------------------------------------------
python3 -B "$W/scripts/build_public_tasks.py" --out "$W/data/public_repo_tasks.jsonl" \
    --count 512 --seed 20261116 --sha256 9a5e4e87b67a53807b0e47eb1019692adf806559f4348281e037a3a387a83c40
