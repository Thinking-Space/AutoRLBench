#!/usr/bin/env bash
# Prepare stage: the only networked stage. One asset is fetched (the base
# model); the training set ships with the workspace (data/), and the vendored
# miles tree is pinned and pre-guarded in-repo (see source_manifest.json).
#
# Runs in a separate prepare-stage container: root, the miles-011 shared
# image, with $AUTORLBENCH_WORKSPACE, $AUTORLBENCH_MODELS and
# $AUTORLBENCH_DATASETS writable.
set -euo pipefail

ENVIRONMENT_DIR="${AUTORLBENCH_ENVIRONMENT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
W="${AUTORLBENCH_WORKSPACE:?prepare requires the staged workspace}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

# --- image-level env, seeded into the prepared handoff -------------------------
for _kv in TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# --- base model (--hf-checkpoint / --ref-load in recipes/run_training.py) ------
# Qwen/Qwen3-0.6B @ c1899de289a04d12100db370d81485cdf75e47ca
#   model.safetensors  1503300328 B  f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
#   config.json               726 B  660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd
# The trainer attention backend is sdpa (not the recipe's stock
# flash_attention_3) because the image's transformers fetches the fa3 kernel
# from the HF hub at runtime and the agent/rerun stages are offline; the
# choice is baked into the recipe and identical in every arm.
python3 "$ENVIRONMENT_DIR/fetch_hf.py" \
    --repo Qwen/Qwen3-0.6B \
    --dest "$W/models/Qwen3-0.6B"

python3 - "$W/models/Qwen3-0.6B" <<'PYEOF'
import hashlib
import sys
from pathlib import Path

model = Path(sys.argv[1])
pins = {
    "model.safetensors": (1503300328, "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"),
    "config.json": (726, "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"),
}
for name, (size, sha) in pins.items():
    path = model / name
    got = path.stat().st_size
    assert got == size, f"{name}: expected {size} bytes, got {got}"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == sha, f"{name}: sha256 mismatch: {digest}"
print("model pin verified: Qwen3-0.6B @ c1899de2")
PYEOF

echo "prepare complete"
