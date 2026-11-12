#!/usr/bin/env bash
# verl_rollout_weight_overwrite — prepare stage.
#
# Runs in the prepare-stage container: root, the vllm0.11-torch2.8-cu128
# image, the
# only networked stage, and $AUTORLBENCH_WORKSPACE writable. It fetches the
# Qwen3-1.7B weights, the DAPO-Math-17k parquet and the AIME24/25 sets, then
# derives the training/eval assets via prepare_data.py, which verifies every
# file against asset_registry.json (the per-file pins live there).
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?The official harness sets this}"
workspace="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"
mkdir -p "$workspace/assets/raw"

# --- image-level env, seeded here --------------------------------------------
# The image bakes no CUDA_DEVICE_ORDER / TOKENIZERS_PARALLELISM defaults; the
# two entries the recording rig's image carried ride the prepared/env
# handoff into the agent/rerun/verifier stages via the activate hook.
# Idempotent.
for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# --- recorded pure-Python pins, shadowed over the image ----------------------
# The image's compiled layer is the recorded one (vLLM 0.11.0 · torch 2.8.0 ·
# flash-attn 2.8.3.post1 · cu128 · py3.12), but its pure-Python layer was
# frozen at different pins. The three that matter here are shadowed from
# $AUTORLBENCH_PREPARED/pysite:
#   * pyarrow 19.0.1 + transformers 4.57.1 — prepare_data.py rebuilds the
#     recorded parquets and hash-checks them against asset_registry.json,
#     and parquet metadata embeds the writer's version, so the image's
#     pyarrow 24.0.0 / transformers 4.57.6 can never reproduce the recorded
#     sha256s (tokenizers is already identical at 0.22.2);
#   * numpy 1.26.4 — the recorded major version; NumPy 2 defaults the NEP 50
#     promotion rules, which can shift mixed-dtype arithmetic during
#     training. torch 2.8.0 + vLLM 0.11.0 + numpy 1.26.4 is the recorded
#     combination, so the shadow is compatible by construction.
# --no-deps: everything else these three need already sits in the image at
# recorded-compatible versions; letting pip resolve the tree would float
# pins the image deliberately carries.
site="$AUTORLBENCH_PREPARED/pysite"
if [ ! -d "$site/pyarrow" ]; then
    _pip_args=(--no-cache-dir --no-deps --only-binary :all: --target "$site")
    if [ -n "${AUTORLBENCH_PIP_INDEX_URL:-}" ]; then
        _pip_args+=(--index-url "$AUTORLBENCH_PIP_INDEX_URL")
        if [ -n "${AUTORLBENCH_PIP_TRUSTED_HOST:-}" ]; then
            _pip_args+=(--trusted-host "$AUTORLBENCH_PIP_TRUSTED_HOST")
        fi
    fi
    python3 -m pip install "${_pip_args[@]}" \
        pyarrow==19.0.1 transformers==4.57.1 numpy==1.26.4
fi
# Shadow for the rest of this script (prepare_data.py must write with the
# recorded pins) and for the agent/rerun/verifier stages via prepared/env.
# The explicit /workspace keeps the image's PYTHONPATH ENV in play — the
# prepared/env line replaces it — so `import verl` still resolves from the
# workspace checkout.
export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
if ! grep -qs "^PYTHONPATH=" "${AUTORLBENCH_PREPARED}/env"; then
    echo "PYTHONPATH=$site:/workspace" >> "${AUTORLBENCH_PREPARED}/env"
fi
PYTHONPATH="$site" python3 - <<'EOF'
import numpy, pyarrow, transformers
assert numpy.__version__ == "1.26.4", numpy.__version__
assert pyarrow.__version__ == "19.0.1", pyarrow.__version__
assert transformers.__version__ == "4.57.1", transformers.__version__
EOF

# The task-local fetch_hf.py fetches the whole repos, cache-first, without
# checksum verification (asset_registry.json is the pin of record); the
# revisions the former autorlbench_fetch_hf allowlist pinned, for reference:
#   Qwen/Qwen3-1.7B @ 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
#   BytedTsinghua-SIA/DAPO-Math-17k @ 61829aa9fe754403c5559e46423772deac5a329b
#   HuggingFaceH4/aime_2024 @ 2fe88a2f1091d5048c0f36abc874fb997b3dd99a
#   yentinglin/aime_2025 @ 6f71d77b0b89b9dabe07ab466c51df33f514df7f
python3 "$env_dir/fetch_hf.py" \
    --repo Qwen/Qwen3-1.7B \
    --dest "$workspace/model"
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo BytedTsinghua-SIA/DAPO-Math-17k \
    --dest "$workspace/assets/raw/dapo"
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo HuggingFaceH4/aime_2024 \
    --dest "$workspace/assets/raw/aime24"
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo yentinglin/aime_2025 \
    --dest "$workspace/assets/raw/aime25"
python "$env_dir/prepare_data.py" --workspace "$workspace" --registry "$env_dir/asset_registry.json"
