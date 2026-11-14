#!/usr/bin/env bash
# Prepare stage: the only networked stage. Fetches the Granite weights and the
# GSM8K parquet files, then rebuilds data/train.jsonl and data/eval.jsonl with
# the workspace's own data_tools.py (full splits, seed-42 order, checksummed).
# The vendored TRL tree is imported from the workspace root by train.py; the
# shared image already provides torch, transformers, datasets and wandb.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"

# ibm-granite/granite-3.3-2b-instruct @ 707f574c62054322f6b5b04b6d075f0a8f05e0f0
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo ibm-granite/granite-3.3-2b-instruct \
    --dest "$W/model"

# openai/gsm8k @ 740312add88f781978c0658806c59bc2815b9866
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --type dataset \
    --repo openai/gsm8k \
    --dest "$W/data/hf"
mkdir -p "$W/data/source"
for split in train test; do
    ln -sf "../hf/main/$split-00000-of-00001.parquet" "$W/data/source/$split-00000-of-00001.parquet"
done
cd "$W"
python3 data_tools.py --output-dir data
