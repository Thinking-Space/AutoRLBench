#!/usr/bin/env bash
# Prepare stage, shared byte-identical by the PiSSA GRPO task pair
# (verl_pissa_residual_base_reference, verl_pissa_vllm_adapter_request).
#
# Runs in the prepare-stage container: root, the vllm0.11-torch2.8-cu128
# image, the only networked stage, and $AUTORLBENCH_WORKSPACE writable. It
# fetches the Qwen2.5-0.5B-Instruct weights and the GSM8K parquet, derives
# the PiSSA rank-8 bundle and the deterministic tiny-GSM8K subsets, then
# pins the derived artifacts by sha256.
set -euo pipefail

W=${AUTORLBENCH_WORKSPACE:-/workspace}
SOURCE="$W/assets/source-qwen2.5-0.5b"
BUNDLE="$W/assets/qwen2.5-0.5b-pissa-r8"
RAW="$W/data/raw-gsm8k"
TINY="$W/data/tiny-gsm8k"

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
# cu128 · py3.12; its extra flash-attn 2.8.3.post1 stays inert — the recorded
# venv had none and the run pins attn_implementation=eager), but its
# pure-Python layer was frozen at different pins. The eight that matter here
# are shadowed from $AUTORLBENCH_PREPARED/pysite:
#   * peft 0.18.0 — PiSSA init (pissa_niter_2) drives the SVD derivation
#     below; any other peft changes the derived bytes;
#   * datasets 4.0.0 + pyarrow 19.0.1 — prepare_tiny_gsm8k.py rebuilds the
#     recorded parquets, and parquet metadata embeds the writer's version;
#   * numpy 1.26.4 — the recorded major version (verl's own requirement is
#     numpy<2.0.0); NumPy 2 defaults the NEP 50 promotion rules, which can
#     shift mixed-dtype arithmetic during training;
#   * scipy 1.17.1 — compiled against numpy>=2, so it must follow numpy into
#     the shadow or the ABI clash breaks transformers' lazy imports;
#   * transformers 4.57.1 · tokenizers 0.22.1 · accelerate 1.11.0 — the
#     recorded model-loading stack.
# Verified 2026-11-10 on real AVX-512 x86 (Xeon Platinum 8468): with exactly
# this shadow, the derivation below reproduces the pins byte-for-byte.
# --no-deps: everything else these need already sits in the image at
# recorded-compatible versions; letting pip resolve the tree would float
# pins the image deliberately carries.
site="$AUTORLBENCH_PREPARED/pysite"
if [ ! -d "$site/peft" ]; then
    _pip_args=(--no-cache-dir --no-deps --only-binary :all: --target "$site")
    if [ -n "${AUTORLBENCH_PIP_INDEX_URL:-}" ]; then
        _pip_args+=(--index-url "$AUTORLBENCH_PIP_INDEX_URL")
        if [ -n "${AUTORLBENCH_PIP_TRUSTED_HOST:-}" ]; then
            _pip_args+=(--trusted-host "$AUTORLBENCH_PIP_TRUSTED_HOST")
        fi
    fi
    python3 -m pip install "${_pip_args[@]}" \
        peft==0.18.0 transformers==4.57.1 datasets==4.0.0 pyarrow==19.0.1 \
        numpy==1.26.4 accelerate==1.11.0 tokenizers==0.22.1 scipy==1.17.1
fi
# Shadow for the rest of this script (the derivations must run on the
# recorded pins) and for the agent/rerun/verifier stages via prepared/env.
# The explicit /workspace keeps the image's PYTHONPATH ENV in play — the
# prepared/env line replaces it — so `import verl` still resolves from the
# workspace checkout.
export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
if ! grep -qs "^PYTHONPATH=" "${AUTORLBENCH_PREPARED}/env"; then
    echo "PYTHONPATH=$site:/workspace" >> "${AUTORLBENCH_PREPARED}/env"
fi
PYTHONPATH="$site" python3 - <<'EOF'
import accelerate, datasets, numpy, peft, pyarrow, scipy, tokenizers, transformers
assert peft.__version__ == "0.18.0", peft.__version__
assert transformers.__version__ == "4.57.1", transformers.__version__
assert datasets.__version__ == "4.0.0", datasets.__version__
assert pyarrow.__version__ == "19.0.1", pyarrow.__version__
assert numpy.__version__ == "1.26.4", numpy.__version__
assert accelerate.__version__ == "1.11.0", accelerate.__version__
assert tokenizers.__version__ == "0.22.1", tokenizers.__version__
assert scipy.__version__ == "1.17.1", scipy.__version__
EOF

# The task-local fetch_hf.py fetches the whole repos, cache-first, without
# checksum verification; the revisions the former autorlbench_fetch_hf
# allowlist pinned are recorded here for reference:
#   Qwen/Qwen2.5-0.5B-Instruct @ 7ae557604adf67be50417f59c2c2f167def9a775
#   openai/gsm8k @ 740312add88f781978c0658806c59bc2815b9866
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen2.5-0.5B-Instruct \
    --dest "$SOURCE"

python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --type dataset \
    --repo openai/gsm8k \
    --dest "$RAW"

python "$AUTORLBENCH_ENVIRONMENT/prepare_bundle.py" --model "$SOURCE" --output "$BUNDLE" --rank 8
python "$AUTORLBENCH_ENVIRONMENT/prepare_tiny_gsm8k.py" --source "$RAW" --output "$TINY" --train-size 16 --val-size 8

# Derived-artifact pins. bf16 SVD (peft pissa_niter_2) and the parquet
# writer footer are build-stable, not cross-env stable, so the pins follow
# the environment that regenerates them byte-exactly. Verified 2026-11-10 on
# the vllm0.11-torch2.8-cu128 image with the pure-Python shadow above, on
# AVX-512 x86 (Xeon Platinum 8468; the derivation is CPU-only):
#   * the two safetensors hashes are the 2026-11-04 re-pins — the author's
#     originals (ee02fd44…, a78b3d4d…) came from a torch-wheel/CPU build
#     that no longer exists, and the re-pins rebuild byte-identically here
#     (the bundle also verifies numerically: W_residual + B@A reconstructs
#     W_pretrained to 2.1e-3 relative, bf16 rounding, across all 48 target
#     modules);
#   * the two parquet hashes are the PR author's originals — the recorded
#     pyarrow 19.0.1 + datasets 4.0.0 combination reproduces them, where
#     the 2026-11-04 rebuild (67a66b20…, d533efb9…) came from an
#     approximate environment.
printf '%s  %s\n' \
  547d56774376499b4e4ee7514c1152894122cab6c277fcdea82a53176ad437f0 "$BUNDLE/residual_base/model.safetensors" \
  2b9b3b62e9619360a84cf18ca4fd45dbbd38b734874a9044d260035f9712a635 "$BUNDLE/pissa_adapter/adapter_model.safetensors" \
  5404367e3311ec89dad8720ab683228999b656b7c611fed8d64963d7bc0c755b "$TINY/train.parquet" \
  01fca7c632b579db57432b02aa732e769a2a33cb3feb2f470c4b37c32500a995 "$TINY/validation.parquet" | sha256sum -c -
