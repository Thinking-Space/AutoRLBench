#!/usr/bin/env bash
# verl_qwen3_vl_grpo_training_case_v8k2 — prepare stage.
#
# Runs in the prepare-stage container: root, the vl-grpo-cu129-recorded
# image, the only networked stage, and $AUTORLBENCH_WORKSPACE writable.
# Recorded assets are immutable: every file is fetched with the task-local
# fetch_hf.py (whole repo, cache-first) and then size+SHA-checked against
# environment/assets.json before use, so an upstream drift fails prepare
# loudly instead of silently rewriting the ground truth. prepare_data.py
# re-verifies the selected rows and the produced parquet outputs.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_ENVIRONMENT:?}"
: "${AUTORLBENCH_PREPARED:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"
export PYTHONDONTWRITEBYTECODE=1

# --- image-level env, seeded here --------------------------------------------
# The recording rig exported thread pinning and bytecode suppression; these
# ride the prepared/env handoff into the agent/rerun/verifier stages via the
# activate hook (WANDB_MODE=offline is exported by the harness itself for
# every stage). Idempotent.
for _kv in PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# prepare_data.py reads/writes parquet and pins outputs by content digest, so
# any pyarrow that round-trips the recorded content passes; only install if the
# image is missing pyarrow entirely.
python3 -c 'import pyarrow' 2>/dev/null \
  || uv pip install --system --python "$(command -v python3)" \
       'pyarrow==21.0.0'

# Whole-repo fetches into the workspace via the task-local fetch tool
# (cache-first, shared across runs); the revisions the former
# autorlbench_fetch_hf allowlist pinned, for reference:
#   Qwen/Qwen3-VL-8B-Instruct @ 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
#   derek-thomas/ScienceQA @ f18b0a70359ebfb41f658fd564208d0355b013f4
python3 "$env_dir/fetch_hf.py" \
    --repo Qwen/Qwen3-VL-8B-Instruct \
    --dest "$W/resources/model"
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo derek-thomas/ScienceQA \
    --dest "$W/resources/raw"

# Size+SHA verification of every pinned file, straight from assets.json (the
# single source of truth for its pins). The fetch pulled the whole repos, so
# this only reads.
python3 -B - "$env_dir/assets.json" "$W" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

pins = json.loads(Path(sys.argv[1]).read_text())
workspace = Path(sys.argv[2])
for asset in pins:
    kind = "model" if asset["kind"] == "model" else "raw"
    for f in asset["files"]:
        if f["path"] == ".gitattributes":
            continue
        path = workspace / "resources" / kind / f["path"]
        if not path.is_file():
            raise SystemExit(f"missing pinned file: {path}")
        if path.stat().st_size != f["bytes"]:
            raise SystemExit(
                f"{path}: expected {f['bytes']} bytes, got {path.stat().st_size}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != f["sha256"]:
            raise SystemExit(f"{path}: sha256 {digest} != pinned {f['sha256']}")
        print(f"[prepare] verified {path.relative_to(workspace)}", flush=True)
PY

python3 "$env_dir/prepare_data.py" \
  --raw-root "$W/resources/raw" \
  --output "$W/resources"
