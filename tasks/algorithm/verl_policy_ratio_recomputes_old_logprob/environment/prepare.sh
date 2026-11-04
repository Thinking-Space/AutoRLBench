#!/usr/bin/env bash
set -euo pipefail
# Fixed subsets of the official GSM8K train/test splits ship with this task.
# environment/data_manifest.json records the pinned upstream files and row selection.
# Every model asset is pinned by revision, byte size, and SHA-256.
autorlbench_fetch_hf \
  --repo Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --dest "$AUTORLBENCH_WORKSPACE/model" \
  --allow config.json \
  --allow generation_config.json \
  --allow model.safetensors \
  --allow tokenizer.json \
  --allow tokenizer_config.json \
  --allow vocab.json \
  --allow merges.txt \
  --allow LICENSE \
  --expect config.json=18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45 --expect-size config.json=659 \
  --expect generation_config.json=e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6 --expect-size generation_config.json=242 \
  --expect model.safetensors=fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe --expect-size model.safetensors=988097824 \
  --expect tokenizer.json=c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 --expect-size tokenizer.json=7031645 \
  --expect tokenizer_config.json=5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583 --expect-size tokenizer_config.json=7305 \
  --expect vocab.json=ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 --expect-size vocab.json=2776833 \
  --expect merges.txt=599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 --expect-size merges.txt=1671839 \
  --expect LICENSE=832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e --expect-size LICENSE=11343

# Reconstruct the packaged selection from immutable official source files.
autorlbench_fetch_hf \
  --repo openai/gsm8k --type dataset \
  --revision 740312add88f781978c0658806c59bc2815b9866 \
  --dest "$AUTORLBENCH_PREPARED/gsm8k-source" \
  --allow main/train-00000-of-00001.parquet \
  --allow main/test-00000-of-00001.parquet \
  --expect main/train-00000-of-00001.parquet=ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d \
  --expect-size main/train-00000-of-00001.parquet=2306545 \
  --expect main/test-00000-of-00001.parquet=ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59 \
  --expect-size main/test-00000-of-00001.parquet=419088
python3 "$AUTORLBENCH_ENVIRONMENT/prepare_data.py" \
  --source "$AUTORLBENCH_PREPARED/gsm8k-source" \
  --output "$AUTORLBENCH_PREPARED/expected-data" --kind algorithm
python3 - "$AUTORLBENCH_PREPARED/expected-data" "$AUTORLBENCH_WORKSPACE/data" <<'PY'
import hashlib
from pathlib import Path
import sys
expected, submitted = map(Path, sys.argv[1:])
for source in expected.iterdir():
    target = submitted/source.name
    assert target.is_file() and hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(target.read_bytes()).digest(), 'packaged data differs from official GSM8K: '+source.name
print('Pinned official GSM8K data verified')
PY
