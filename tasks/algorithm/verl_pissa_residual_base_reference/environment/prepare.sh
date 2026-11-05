#!/usr/bin/env bash
# Prepare stage, shared byte-identical by the verl-vllm011 task pair
# (verl_pissa_residual_base_reference, verl_pissa_vllm_adapter_request).
#
# Runs in the prepare-stage container: root, the verl-vllm011 image, the only
# networked stage, and $AUTORLBENCH_WORKSPACE writable. It fetches the
# Qwen2.5-0.5B-Instruct weights and the GSM8K parquet, derives the PiSSA
# rank-8 bundle and the deterministic tiny-GSM8K subsets, then pins the
# derived artifacts by sha256.
set -euo pipefail

W=${AUTORLBENCH_WORKSPACE:-/workspace}
SOURCE="$W/assets/source-qwen2.5-0.5b"
BUNDLE="$W/assets/qwen2.5-0.5b-pissa-r8"
RAW="$W/data/raw-gsm8k"
TINY="$W/data/tiny-gsm8k"

# --- image-level env, seeded here --------------------------------------------
# docker/images/verl-vllm011.Dockerfile is a build recipe baked without ENV;
# the two entries the recording rig's image carried ride the prepared/env
# handoff into the agent/rerun/verifier stages via the activate hook.
# Idempotent.
for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

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

# Derived-artifact pins: the PR author's originals (ee02fd44…, a78b3d4d…,
# 5404367e…, 01fca7c6…) do not byte-reproduce outside their build env
# (bf16 SVD + parquet writer footer are build-stable, not cross-env stable).
# Re-pinned 2026-11-04 to a deterministic rebuild in the task's recorded
# build environment: two consecutive builds agree, and the bundle verifies
# numerically (W_residual + B@A reconstructs W_pretrained to 2.1e-3 relative,
# bf16 rounding, across all 48 target modules). Source pins above unchanged.
printf '%s  %s\n' \
  547d56774376499b4e4ee7514c1152894122cab6c277fcdea82a53176ad437f0 "$BUNDLE/residual_base/model.safetensors" \
  2b9b3b62e9619360a84cf18ca4fd45dbbd38b734874a9044d260035f9712a635 "$BUNDLE/pissa_adapter/adapter_model.safetensors" \
  67a66b2052107c42b179a9bd2be200edd70a12857ebfdbede6ee07b53f3f9cbc "$TINY/train.parquet" \
  d533efb9150ef490c42058cc161ab45e310fe7108a951e365d47d98a1fc1a5d4 "$TINY/validation.parquet" | sha256sum -c -
