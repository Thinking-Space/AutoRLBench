#!/usr/bin/env bash
# verl_lora_merge_base_weight_sync: prepare stage.
#
# Runs in a separate prepare-stage container: root, the `verl` image, **the only networked stage**, and
# $AUTORLBENCH_WORKSPACE is writable. It fetches the pinned model and GSM8K, then builds the parquet files
# with verl's own examples/data_preprocess/gsm8k.py from the workspace checkout, exactly as the recorded runs did.
#
# Destinations are what run_grpo_lora.sh reads: MODEL=$WS/models/Qwen2.5-1.5B-Instruct and
# DATA=$WS/data/gsm8k/{train,test}.parquet.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
mkdir -p "$W/.tmp" "$W/.home"
export HOME="$W/.home" TMPDIR="$W/.tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

# --- Base model ---------------------------------------------------------------
autorlbench_fetch_hf \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --type model \
    --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    --allow LICENSE \
    --expect LICENSE=832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e \
    --expect-size LICENSE=11343 \
    --allow README.md \
    --expect README.md=2e1bcd8bd964728a820be709fa0f7b9dd54817a94fd2254c535df70c5e67fada \
    --expect-size README.md=4917 \
    --allow config.json \
    --expect config.json=98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670 \
    --expect-size config.json=660 \
    --allow generation_config.json \
    --expect generation_config.json=e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6 \
    --expect-size generation_config.json=242 \
    --allow merges.txt \
    --expect merges.txt=599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 \
    --expect-size merges.txt=1671839 \
    --allow model.safetensors \
    --expect model.safetensors=dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee \
    --expect-size model.safetensors=3087467144 \
    --allow tokenizer.json \
    --expect tokenizer.json=c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 \
    --expect-size tokenizer.json=7031645 \
    --allow tokenizer_config.json \
    --expect tokenizer_config.json=5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583 \
    --expect-size tokenizer_config.json=7305 \
    --allow vocab.json \
    --expect vocab.json=ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 \
    --expect-size vocab.json=2776833 \
    --dest "$W/models/Qwen2.5-1.5B-Instruct"

# --- Dataset ------------------------------------------------------------------
autorlbench_fetch_hf \
    --repo openai/gsm8k \
    --type dataset \
    --revision 740312add88f781978c0658806c59bc2815b9866 \
    --allow README.md \
    --expect README.md=a17e882503578c9e324560630e31017617714ee025c87cf4fea6fd916895f3c1 \
    --expect-size README.md=7935 \
    --allow main/train-00000-of-00001.parquet \
    --expect main/train-00000-of-00001.parquet=ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d \
    --expect-size main/train-00000-of-00001.parquet=2306545 \
    --allow main/test-00000-of-00001.parquet \
    --expect main/test-00000-of-00001.parquet=ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59 \
    --expect-size main/test-00000-of-00001.parquet=419088 \
    --dest "$W/data/raw/gsm8k"

if [ ! -f "$W/data/gsm8k/.complete" ]; then
    mkdir -p "$W/data/gsm8k"
    PYTHONPATH="$W/verl" python3 "$W/verl/examples/data_preprocess/gsm8k.py" \
        --local_dataset_path "$W/data/raw/gsm8k" --local_save_dir "$W/data/gsm8k"
    # Byte-identical to the parquet files the recorded runs trained and validated on (7473 / 1319 rows).
    (cd "$W/data/gsm8k" && sha256sum -c - <<'EOF'
8fb13a0cd8621e5fcacdee3ffe28b3a369975478ea7ab514781f3f190cba2488  train.parquet
0dcd50ed32caa9d8f434d4e3ef8031bc3e49b0f823655a53be5c002e727f17dd  test.parquet
EOF
    )
    touch "$W/data/gsm8k/.complete"
fi
