#!/usr/bin/env bash
# Recorded assets are immutable. Every downloaded file is checked before conversion.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
export W="$AUTORLBENCH_WORKSPACE"
mkdir -p "$W/metadata" "$W/data/gsm8k" "$W/.tmp" "$W/.home"
export HOME="$W/.home" TMPDIR="$W/.tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$W/src/$(cat "$W/variant.txt"):$W/scripts"
autorlbench_fetch_hf \
    --repo Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --type model \
    --revision ec052fda178e241c7c443468d2fa1db6618996be \
    --allow LICENSE \
    --expect LICENSE=6e521a32ba6661de2b3e7b1cd27d7e1e9b8ff99633c79462a23284cac13c8b8d \
    --expect-size LICENSE=6897 \
    --allow README.md \
    --expect README.md=e495b6b36fcd5bb868e989bc193a13416d20a2ef77c8168069e5869919981141 \
    --expect-size README.md=2798 \
    --allow config.json \
    --expect config.json=b3e9fc1ad643d12f6536dee80a66c5ea045b571856165a598252720b6f2c553e \
    --expect-size config.json=920 \
    --allow configuration.json \
    --expect configuration.json=f625873f5df26193fae3ff45d3054667f5a9e91d644156add3310f258c495383 \
    --expect-size configuration.json=38 \
    --allow generation_config.json \
    --expect generation_config.json=bb1ace90ba308b9e8b3c823849aca96c3428773c715b38493ebcd9ccef4c6cac \
    --expect-size generation_config.json=248 \
    --allow merges.txt \
    --expect merges.txt=599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 \
    --expect-size merges.txt=1671839 \
    --allow model-00001-of-00008.safetensors \
    --expect model-00001-of-00008.safetensors=a51a84c031e0f6c8808bc284106c488ab3db6069a3f6f5a231be1e963206d020 \
    --expect-size model-00001-of-00008.safetensors=3999614664 \
    --allow model-00002-of-00008.safetensors \
    --expect model-00002-of-00008.safetensors=02a9f530c526ac09e5d26cdece113d62b030ebc6f137ea69cd2754bea1287121 \
    --expect-size model-00002-of-00008.safetensors=3999385424 \
    --allow model-00003-of-00008.safetensors \
    --expect model-00003-of-00008.safetensors=200116fdf90ea17c2dae53594a87b9efbbe199e4c4e487fb25aba9c65e674112 \
    --expect-size model-00003-of-00008.safetensors=3988628968 \
    --allow model-00004-of-00008.safetensors \
    --expect model-00004-of-00008.safetensors=41296f5c77188a4182c87f9bc5958222897ed5fe1bde6912755df050f1222b8d \
    --expect-size model-00004-of-00008.safetensors=3999386088 \
    --allow model-00005-of-00008.safetensors \
    --expect model-00005-of-00008.safetensors=a613a2c68f18630beb1eb681627d34110cec36a615a35c4d7f35f72e0df5b2e4 \
    --expect-size model-00005-of-00008.safetensors=3988629648 \
    --allow model-00006-of-00008.safetensors \
    --expect model-00006-of-00008.safetensors=44de056d0f2c4375685ec7545a987c20d59cd8ee75c1802e04a11e7dcd7a84f1 \
    --expect-size model-00006-of-00008.safetensors=3999386104 \
    --allow model-00007-of-00008.safetensors \
    --expect model-00007-of-00008.safetensors=fbc95c8fc0842c425168c21e933629cde22fc156ef9b4af927ad2787fe4d33a1 \
    --expect-size model-00007-of-00008.safetensors=3988629648 \
    --allow model-00008-of-00008.safetensors \
    --expect model-00008-of-00008.safetensors=9a5e51c0a101292c249032382f20d912677e1c0c4b5b0c78e951450fedb820ee \
    --expect-size model-00008-of-00008.safetensors=668484400 \
    --allow model.safetensors.index.json \
    --expect model.safetensors.index.json=ece1b223efe32f4349d0dfa2a522249ac10bcb89369ed25b222c35175cd90b53 \
    --expect-size model.safetensors.index.json=416452 \
    --allow tokenizer.json \
    --expect tokenizer.json=f7c9b2dba4a296b1aa76c16a34b8225c0c118978400d4bb66bff0902d702f5b8 \
    --expect-size tokenizer.json=7028015 \
    --allow tokenizer_config.json \
    --expect tokenizer_config.json=3ec7934124b5e81aaa9c0977b95299b8ca3911b91bf8114494023fbbb22a515f \
    --expect-size tokenizer_config.json=1289 \
    --allow vocab.json \
    --expect vocab.json=ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 \
    --expect-size vocab.json=2776833 \
    --dest "$W/models/Qwen1.5-MoE-A2.7B-Chat"

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
    --allow README.md \
    --expect README.md=a17e882503578c9e324560630e31017617714ee025c87cf4fea6fd916895f3c1 \
    --expect-size README.md=7935 \
    --dest "$W/data/raw/gsm8k"

python3 "$W/scripts/prepare_gsm8k.py"
