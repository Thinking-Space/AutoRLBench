#!/usr/bin/env bash
set -euo pipefail

autorlbench_fetch_hf \
  --repo Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --allow config.json \
  --expect config.json=18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45 \
  --expect-size config.json=659 \
  --allow generation_config.json \
  --expect generation_config.json=e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6 \
  --expect-size generation_config.json=242 \
  --allow tokenizer.json \
  --expect tokenizer.json=c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 \
  --expect-size tokenizer.json=7031645 \
  --allow tokenizer_config.json \
  --expect tokenizer_config.json=5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583 \
  --expect-size tokenizer_config.json=7305 \
  --allow merges.txt \
  --expect merges.txt=599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 \
  --expect-size merges.txt=1671839 \
  --allow vocab.json \
  --expect vocab.json=ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 \
  --expect-size vocab.json=2776833 \
  --allow model.safetensors \
  --expect model.safetensors=fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe \
  --expect-size model.safetensors=988097824 \
  --allow LICENSE \
  --expect LICENSE=832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e \
  --expect-size LICENSE=11343 \
  --dest "$AUTORLBENCH_WORKSPACE/model"

autorlbench_fetch_hf \
  --repo openai/gsm8k --type dataset \
  --revision 740312add88f781978c0658806c59bc2815b9866 \
  --allow main/train-00000-of-00001.parquet \
  --expect main/train-00000-of-00001.parquet=ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d \
  --expect-size main/train-00000-of-00001.parquet=2306545 \
  --allow main/test-00000-of-00001.parquet \
  --expect main/test-00000-of-00001.parquet=ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59 \
  --expect-size main/test-00000-of-00001.parquet=419088 \
  --dest "$AUTORLBENCH_WORKSPACE/data"
ln -sf main/train-00000-of-00001.parquet "$AUTORLBENCH_WORKSPACE/data/train.parquet"
ln -sf main/test-00000-of-00001.parquet "$AUTORLBENCH_WORKSPACE/data/test.parquet"

python3 -m pip install --no-deps --no-build-isolation "$AUTORLBENCH_WORKSPACE"
