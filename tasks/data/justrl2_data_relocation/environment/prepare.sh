#!/usr/bin/env bash
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
workspace="$AUTORLBENCH_WORKSPACE"
cd "$workspace"
if [[ -f prepared/.complete ]]; then
  python3 -B -c 'from prepare_assets import verify_prepared; verify_prepared("prepared")'
  python3 -B -c 'from prepare_assets import expose_framework_source; expose_framework_source("prepared", "framework")'
  exit 0
fi
mkdir -p "$workspace/cache" "$workspace/public-source"

autorlbench_fetch_hf \
  --repo openbmb/JustRL-II-base-model \
  --revision b558bbc71d9433e7c5869effcbac07d4c7f2c5b4 \
  --allow README.md \
  --expect README.md=131cd28701349bc98bb98b0a3583ac69534b57a05130b7c1c7d4e69fd8ce335c \
  --expect-size README.md=5097 \
  --allow chat_template.jinja \
  --expect chat_template.jinja=e30b4fff39640a56a795c407be00b131b2dbaf3bd040ae5e8201ea3ffa28bb4c \
  --expect-size chat_template.jinja=9076 \
  --allow config.json \
  --expect config.json=0f8ea3cf839f92d8a0f15299856136cfc5c9150d849fa907425deea36c1100d9 \
  --expect-size config.json=624 \
  --allow pytorch_model.bin \
  --expect pytorch_model.bin=8d02154edd2c2cdaff290c286c79342162812f8564d65302011241104c53fb99 \
  --expect-size pytorch_model.bin=5033637787 \
  --allow special_tokens_map.json \
  --expect special_tokens_map.json=82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db \
  --expect-size special_tokens_map.json=551 \
  --allow tokenizer.json \
  --expect tokenizer.json=3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81 \
  --expect-size tokenizer.json=9894271 \
  --allow tokenizer_config.json \
  --expect tokenizer_config.json=094efb3cf1ff412284cc5945fc99dff58673a912760d04483a04aa1c716f66fd \
  --expect-size tokenizer_config.json=94416 \
  --dest "$workspace/cache/model"

autorlbench_fetch_url \
  --url https://codeload.github.com/merak0514/JustRL-II/tar.gz/75814318f243af7e39af337c1173fc3927c9635b \
  --sha256 09e7548e98bb3285526eb7582d06b5e671165cec7fab98ae7876081def5ed9b5 --size 493591 \
  --expect source.tar.gz=09e7548e98bb3285526eb7582d06b5e671165cec7fab98ae7876081def5ed9b5 \
  --expect-size source.tar.gz=493591 \
  --dest "$workspace/cache/source.tar.gz"

autorlbench_fetch_url \
  --url https://huggingface.co/datasets/openbmb/UltraData-RL-2609/resolve/e6ecfa733708a4c54b5a98c3ca0fd16fc6923790/data/Math/Math_part-1-of-4.jsonl \
  --sha256 7253d74d827b0c357fef12646a4c247b6894e95adafd56cdae67b713d4220150 --size 4803619 \
  --expect Math_part-1-of-4.jsonl=7253d74d827b0c357fef12646a4c247b6894e95adafd56cdae67b713d4220150 \
  --expect-size Math_part-1-of-4.jsonl=4803619 \
  --dest "$workspace/public-source/Math_part-1-of-4.jsonl"

autorlbench_fetch_url \
  --url https://huggingface.co/datasets/openbmb/UltraData-RL-2609/resolve/e6ecfa733708a4c54b5a98c3ca0fd16fc6923790/data/Math/Math_part-2-of-4.jsonl \
  --sha256 33a2aaca39443504dff81e234421308ebbb3135627a1dfa5501e3487f4b69f7f --size 4182093 \
  --expect Math_part-2-of-4.jsonl=33a2aaca39443504dff81e234421308ebbb3135627a1dfa5501e3487f4b69f7f \
  --expect-size Math_part-2-of-4.jsonl=4182093 \
  --dest "$workspace/public-source/Math_part-2-of-4.jsonl"

autorlbench_fetch_url \
  --url https://huggingface.co/datasets/openbmb/UltraData-RL-2609/resolve/e6ecfa733708a4c54b5a98c3ca0fd16fc6923790/data/Math/Math_part-3-of-4.jsonl \
  --sha256 778ecc93e229f2a1c3197561c977e87f86afe630383bf1b3779adf05c07e5d70 --size 4387044 \
  --expect Math_part-3-of-4.jsonl=778ecc93e229f2a1c3197561c977e87f86afe630383bf1b3779adf05c07e5d70 \
  --expect-size Math_part-3-of-4.jsonl=4387044 \
  --dest "$workspace/public-source/Math_part-3-of-4.jsonl"

autorlbench_fetch_url \
  --url https://huggingface.co/datasets/openbmb/UltraData-RL-2609/resolve/e6ecfa733708a4c54b5a98c3ca0fd16fc6923790/data/Math/Math_part-4-of-4.jsonl \
  --sha256 1838dd71bf537ff8a5d0ed87aa11b43d4ef07220a21b5a759a0d74b57f0e289e --size 1233160 \
  --expect Math_part-4-of-4.jsonl=1838dd71bf537ff8a5d0ed87aa11b43d4ef07220a21b5a759a0d74b57f0e289e \
  --expect-size Math_part-4-of-4.jsonl=1233160 \
  --dest "$workspace/public-source/Math_part-4-of-4.jsonl"

python3 -B prepare_assets.py --asset-dir "$workspace/cache" \
  --source-dir "$workspace/public-source" --out "$workspace/prepared"
python3 -B -c 'from prepare_assets import expose_framework_source; expose_framework_source("prepared", "framework")'
touch "$workspace/prepared/.complete"
