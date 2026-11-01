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

printf '%s  %s\n' \
  ee02fd449dd9d78f2d6dcd2524786b9bcc8749ab1505856185baf0379426701e "$BUNDLE/residual_base/model.safetensors" \
  a78b3d4d6ead3a5c06db20bf739ce7749258d1848083c867ea9842a7035f6f35 "$BUNDLE/pissa_adapter/adapter_model.safetensors" \
  5404367e3311ec89dad8720ab683228999b656b7c611fed8d64963d7bc0c755b "$TINY/train.parquet" \
  01fca7c632b579db57432b02aa732e769a2a33cb3feb2f470c4b37c32500a995 "$TINY/validation.parquet" | sha256sum -c -
