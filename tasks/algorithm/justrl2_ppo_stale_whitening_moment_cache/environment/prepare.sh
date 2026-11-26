#!/usr/bin/env bash
# AutoRLBench prepare stage for JustRL-II Countdown PPO tasks.
#
# Runs in the prepare-stage container (the only networked stage) with
# $AUTORLBENCH_WORKSPACE writable. Fetches pinned HuggingFace assets with
# SHA-256 + byte-size verification, converts pytorch_model.bin -> model.safetensors
# on CPU for SGLang 0.5.20 compatibility (then removes raw .bin to save disk),
# builds disjoint Countdown JSONL splits, and converts HF -> Megatron torch_dist
# when CUDA is available (otherwise run_justrl2_ppo.sh converts on first GPU launch).
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_ENVIRONMENT:?}"
: "${AUTORLBENCH_PREPARED:?}"

W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"

for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# --- Base model ---------------------------------------------------------------
# Pinned HF revision: openbmb/JustRL-II-base-model @ b558bbc71d9433e7c5869effcbac07d4c7f2c5b4
RAW_MODEL="$W/models/JustRL-II-base-model-raw"
HF_MODEL="$W/models/JustRL-II-base-model"
MEGATRON_MODEL="$W/models/JustRL-II-base-model-torch_dist"

if [ ! -f "$HF_MODEL/.complete" ]; then
    python3 "$env_dir/autorlbench_fetch.py" hf \
        --repo openbmb/JustRL-II-base-model \
        --revision b558bbc71d9433e7c5869effcbac07d4c7f2c5b4 \
        --allow README.md \
        --allow chat_template.jinja \
        --allow config.json \
        --allow pytorch_model.bin \
        --allow special_tokens_map.json \
        --allow tokenizer.json \
        --allow tokenizer_config.json \
        --expect README.md=131cd28701349bc98bb98b0a3583ac69534b57a05130b7c1c7d4e69fd8ce335c \
        --expect-size README.md=5097 \
        --expect chat_template.jinja=e30b4fff39640a56a795c407be00b131b2dbaf3bd040ae5e8201ea3ffa28bb4c \
        --expect-size chat_template.jinja=9076 \
        --expect config.json=0f8ea3cf839f92d8a0f15299856136cfc5c9150d849fa907425deea36c1100d9 \
        --expect-size config.json=624 \
        --expect pytorch_model.bin=8d02154edd2c2cdaff290c286c79342162812f8564d65302011241104c53fb99 \
        --expect-size pytorch_model.bin=5033637787 \
        --expect special_tokens_map.json=82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db \
        --expect-size special_tokens_map.json=551 \
        --expect tokenizer.json=3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81 \
        --expect-size tokenizer.json=9894271 \
        --expect tokenizer_config.json=094efb3cf1ff412284cc5945fc99dff58673a912760d04483a04aa1c716f66fd \
        --expect-size tokenizer_config.json=94416 \
        --dest "$RAW_MODEL"

    mkdir -p "$HF_MODEL"
    python3 - "$RAW_MODEL" "$HF_MODEL" <<'PY'
import shutil, sys
from pathlib import Path
import torch
from safetensors.torch import save_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for p in src.iterdir():
    if p.name == "pytorch_model.bin" or not p.is_file():
        continue
    shutil.copy2(p, dst / p.name)
state = torch.load(src / "pytorch_model.bin", map_location="cpu", weights_only=True)
save_file({k: v.contiguous() for k, v in state.items()}, str(dst / "model.safetensors"))
(dst / ".complete").write_text("ok\n", encoding="utf-8")
print(f"Converted {len(state)} tensors -> {dst / 'model.safetensors'}")
PY
    rm -rf "$RAW_MODEL"
fi

# --- Dataset ------------------------------------------------------------------
# Pinned HF revision: Jiayi-Pan/Countdown-Tasks-3to4 @ 408f70d177020686d34a56bba5952feb45aaaee4
python3 "$env_dir/autorlbench_fetch.py" hf \
    --repo Jiayi-Pan/Countdown-Tasks-3to4 \
    --type dataset \
    --revision 408f70d177020686d34a56bba5952feb45aaaee4 \
    --allow README.md \
    --allow data/train-00000-of-00001.parquet \
    --expect README.md=eb56eda2e272457a5c87af03492ca62b966db69117bd6e172d8f44c93b79c373 \
    --expect-size README.md=314 \
    --expect data/train-00000-of-00001.parquet=a1b37bea0f439c858b11b908daa682e521f13f9592f1866b08650f58b730d126 \
    --expect-size data/train-00000-of-00001.parquet=2845904 \
    --dest "$W/raw/countdown"

if [ ! -f "$W/data/countdown/.complete" ]; then
    python3 "$W/data_prep/countdown.py" --src "$W/raw/countdown" --out "$W/data/countdown"
    touch "$W/data/countdown/.complete"
fi

# --- Optional Megatron torch_dist conversion (only when CUDA is present) ------
if [ ! -f "$MEGATRON_MODEL/latest_checkpointed_iteration.txt" ]; then
    if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        mkdir -p "$MEGATRON_MODEL"
        (
            cd "$W"
            source justrl2/model_args/minicpm5-2b.sh
            HF_TRUST_REMOTE_CODE=1 \
            CUDA_DEVICE_MAX_CONNECTIONS=1 \
            NVTE_FUSED_ATTN=1 \
            NVTE_FLASH_ATTN=0 \
            NVTE_UNFUSED_ATTN=0 \
            WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29365 \
            PYTHONPATH="$W:/root/Megatron-LM:/sgl-workspace/sglang/python${PYTHONPATH:+:$PYTHONPATH}" \
                python3 tools/convert_hf_to_torch_dist.py \
                    "${MODEL_ARGS[@]}" \
                    --attention-backend fused \
                    --no-gradient-accumulation-fusion \
                    --hf-checkpoint "$HF_MODEL" \
                    --save "$MEGATRON_MODEL"
        )
    else
        echo "No CUDA device in prepare stage; run_justrl2_ppo.sh will convert HF -> torch_dist on first GPU start."
    fi
fi
