#!/usr/bin/env bash
set -euo pipefail
autorlbench_fetch_hf \
    --repo Qwen/Qwen3-4B \
    --revision 1cfa9a7208912126459214e8b04321603b3df60c \
    --allow .gitattributes \
    --expect .gitattributes=34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930 \
    --expect-size .gitattributes=1570 \
    --allow LICENSE \
    --expect LICENSE=832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e \
    --expect-size LICENSE=11343 \
    --allow README.md \
    --expect README.md=71add1cd091c309b1cb1a6b943f2476eb2b004124b3bd1d32dcd0b24cc1abc47 \
    --expect-size README.md=16857 \
    --allow config.json \
    --expect config.json=8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a \
    --expect-size config.json=726 \
    --allow generation_config.json \
    --expect generation_config.json=2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2 \
    --expect-size generation_config.json=239 \
    --allow merges.txt \
    --expect merges.txt=8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5 \
    --expect-size merges.txt=1671853 \
    --allow model-00001-of-00003.safetensors \
    --expect model-00001-of-00003.safetensors=328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223 \
    --expect-size model-00001-of-00003.safetensors=3957900840 \
    --allow model-00002-of-00003.safetensors \
    --expect model-00002-of-00003.safetensors=6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5 \
    --expect-size model-00002-of-00003.safetensors=3987450520 \
    --allow model-00003-of-00003.safetensors \
    --expect model-00003-of-00003.safetensors=e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1 \
    --expect-size model-00003-of-00003.safetensors=99630640 \
    --allow model.safetensors.index.json \
    --expect model.safetensors.index.json=6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2 \
    --expect-size model.safetensors.index.json=32819 \
    --allow tokenizer.json \
    --expect tokenizer.json=aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 \
    --expect-size tokenizer.json=11422654 \
    --allow tokenizer_config.json \
    --expect tokenizer_config.json=d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101 \
    --expect-size tokenizer_config.json=9732 \
    --allow vocab.json \
    --expect vocab.json=ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 \
    --expect-size vocab.json=2776833 \
    --dest "$AUTORLBENCH_WORKSPACE/assets/model"
python -m pip install --no-deps --no-build-isolation -e "$AUTORLBENCH_WORKSPACE/vendor/verl"
python -m compileall -q "$AUTORLBENCH_WORKSPACE/reproductions"
cd "$AUTORLBENCH_WORKSPACE"
python -m reproductions.sql_credit.workload --output data/sql
