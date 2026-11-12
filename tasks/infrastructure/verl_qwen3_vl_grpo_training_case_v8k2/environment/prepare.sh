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

# --- recorded numpy, shadowed over the image ---------------------------------
# The cu128 image's compiled layer matches the recorded stack version-for-
# version (vLLM 0.11.0 / torch 2.8.0 / triton 3.4.0 / xformers 0.0.32.post1 /
# transformers 4.57.6 / nccl 2.27.3 / cudnn 9.10.2.21); the differences are
# the CUDA build tag (cu128 vs cu129 wheels of torch/vLLM/flash-attn), ray
# 2.56.0 vs the recorded 2.49.2, and pure-Python pins. Of those pins only the
# numpy major is load-bearing here: the vendored verl pins numpy<2.0.0
# (framework/setup.py), while the image froze numpy 2.2.6 — NumPy 2's NEP 50
# promotion rules can shift mixed-dtype arithmetic during training. scipy
# must follow numpy into the shadow (its image build is compiled against
# numpy>=2; the ABI clash would break transformers' lazy imports).
# --no-deps: everything else these two need already sits in the image at
# recorded-compatible versions. The same numpy 1.26.4 + scipy 1.17.1 pair was
# verified on this image on real AVX-512 x86 (PiSSA task pair, 2026-11-10).
site="$AUTORLBENCH_PREPARED/pysite"
if [ ! -d "$site/numpy" ]; then
    _pip_args=(--no-cache-dir --no-deps --only-binary :all: --target "$site")
    if [ -n "${AUTORLBENCH_PIP_INDEX_URL:-}" ]; then
        _pip_args+=(--index-url "$AUTORLBENCH_PIP_INDEX_URL")
        if [ -n "${AUTORLBENCH_PIP_TRUSTED_HOST:-}" ]; then
            _pip_args+=(--trusted-host "$AUTORLBENCH_PIP_TRUSTED_HOST")
        fi
    fi
    python3 -m pip install "${_pip_args[@]}" numpy==1.26.4 scipy==1.17.1
fi
# Shadow for the rest of this script and for the agent/rerun/verifier stages
# via prepared/env. The explicit /workspace keeps the image's PYTHONPATH ENV
# in play — the prepared/env line replaces it.
export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
if ! grep -qs "^PYTHONPATH=" "${AUTORLBENCH_PREPARED}/env"; then
    echo "PYTHONPATH=$site:/workspace" >> "${AUTORLBENCH_PREPARED}/env"
fi
PYTHONPATH="$site" python3 - <<'EOF'
import numpy, scipy
assert numpy.__version__ == "1.26.4", numpy.__version__
assert scipy.__version__ == "1.17.1", scipy.__version__
EOF

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
