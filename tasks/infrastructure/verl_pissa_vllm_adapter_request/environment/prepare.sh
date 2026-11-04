#!/usr/bin/env bash
set -euo pipefail

W=${AUTORLBENCH_WORKSPACE:-/workspace}
SOURCE="$W/assets/source-qwen2.5-0.5b"
BUNDLE="$W/assets/qwen2.5-0.5b-pissa-r8"
RAW="$W/data/raw-gsm8k"
TINY="$W/data/tiny-gsm8k"

autorlbench_fetch_hf \
  --repo Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --allow config.json --allow generation_config.json --allow merges.txt \
  --allow tokenizer.json --allow tokenizer_config.json --allow vocab.json \
  --allow model.safetensors \
  --expect model.safetensors=fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe \
  --expect-size model.safetensors=988097824 \
  --dest "$SOURCE"

autorlbench_fetch_hf \
  --repo openai/gsm8k --type dataset \
  --revision 740312add88f781978c0658806c59bc2815b9866 \
  --allow main/train-00000-of-00001.parquet \
  --expect main/train-00000-of-00001.parquet=ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d \
  --expect-size main/train-00000-of-00001.parquet=2306545 \
  --allow main/test-00000-of-00001.parquet \
  --expect main/test-00000-of-00001.parquet=ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59 \
  --expect-size main/test-00000-of-00001.parquet=419088 \
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
