#!/usr/bin/env bash
# verl_gsm8k_flat_eval — prepare stage.
#
# Runs in a separate prepare-stage container: root, `verl` image, **the only
# stage with network**. In the agent stage this file is mounted read-only at
# /autorlbench/environment.
#
# Recorded assets are immutable. Every downloaded file is size+SHA checked
# against the pins below before use; prepare_data.py re-verifies the derived
# GSM8K rows against content pins (writer-independent).
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
export PYTHONDONTWRITEBYTECODE=1

# prepare_data.py reads/writes parquet. The verl image ships pyarrow; pin the
# recorded major version if a rebuilt image is ever missing it.
python3 -c 'import pyarrow' 2>/dev/null \
  || uv pip install --system --python "$(command -v python3)" 'pyarrow==20.0.0'

autorlbench_fetch_hf \
    --repo Qwen/Qwen3-1.7B \
    --type model \
    --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --allow LICENSE \
    --expect LICENSE=832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e \
    --expect-size LICENSE=11343 \
    --allow config.json \
    --expect config.json=1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197 \
    --expect-size config.json=726 \
    --allow generation_config.json \
    --expect generation_config.json=2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2 \
    --expect-size generation_config.json=239 \
    --allow merges.txt \
    --expect merges.txt=8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5 \
    --expect-size merges.txt=1671853 \
    --allow model.safetensors.index.json \
    --expect model.safetensors.index.json=0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d \
    --expect-size model.safetensors.index.json=25605 \
    --allow model-00001-of-00002.safetensors \
    --expect model-00001-of-00002.safetensors=169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5 \
    --expect-size model-00001-of-00002.safetensors=3441185608 \
    --allow model-00002-of-00002.safetensors \
    --expect model-00002-of-00002.safetensors=912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9 \
    --expect-size model-00002-of-00002.safetensors=622329984 \
    --allow tokenizer.json \
    --expect tokenizer.json=aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 \
    --expect-size tokenizer.json=11422654 \
    --allow tokenizer_config.json \
    --expect tokenizer_config.json=d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101 \
    --expect-size tokenizer_config.json=9732 \
    --allow vocab.json \
    --expect vocab.json=ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 \
    --expect-size vocab.json=2776833 \
    --dest "$W/assets/Qwen3-1.7B"

autorlbench_fetch_hf \
    --repo openai/gsm8k \
    --type dataset \
    --revision 740312add88f781978c0658806c59bc2815b9866 \
    --allow main/train-00000-of-00001.parquet \
    --expect main/train-00000-of-00001.parquet=ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d \
    --expect-size main/train-00000-of-00001.parquet=2306545 \
    --allow main/test-00000-of-00001.parquet \
    --expect main/test-00000-of-00001.parquet=ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59 \
    --expect-size main/test-00000-of-00001.parquet=419088 \
    --dest "$W/assets/gsm8k_raw"

# Transform the pinned raw rows into VERL's five-column schema; the output
# content is re-verified against pinned digests (7473 train / 1319 test rows).
python3 "$AUTORLBENCH_ENVIRONMENT/prepare_data.py" \
    --raw-dir "$W/assets/gsm8k_raw" \
    --output "$W/assets/gsm8k"
