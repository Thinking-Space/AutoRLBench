#!/usr/bin/env bash
# verl_lora_merge_base_weight_sync: prepare stage.
#
# Runs in a separate prepare-stage container: root, the
# verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12 image, **the only
# networked stage**, and $AUTORLBENCH_WORKSPACE is writable. It fetches the
# model and GSM8K, then builds the parquet files with verl's own
# examples/data_preprocess/gsm8k.py from the workspace checkout, exactly as
# the recorded runs did.
#
# Destinations are what run_grpo_lora.sh reads: MODEL=$WS/models/Qwen2.5-1.5B-Instruct and
# DATA=$WS/data/gsm8k/{train,test}.parquet.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"
mkdir -p "$W/.tmp" "$W/.home"
export HOME="$W/.home" TMPDIR="$W/.tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

# --- image-level env, seeded here --------------------------------------------
# The retired `verl` recipe baked CUDA_DEVICE_ORDER/TOKENIZERS_PARALLELISM as
# image ENV; those two entries ride the prepared/env handoff into the
# agent/rerun/verifier stages via the activate hook. WANDB_MODE/WANDB_API_KEY
# are exported by the harness itself for every stage. Idempotent.
for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# The task-local fetch_hf.py fetches the whole repos, cache-first, without
# checksum verification; the revisions (and per-file pins) the former
# autorlbench_fetch_hf allowlist recorded, for reference:
#   Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
#   openai/gsm8k @ 740312add88f781978c0658806c59bc2815b9866
python3 "$env_dir/fetch_hf.py" \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --dest "$W/models/Qwen2.5-1.5B-Instruct"
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo openai/gsm8k \
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
