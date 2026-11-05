#!/usr/bin/env bash
set -euo pipefail

autorlbench_fetch_hf \
  --repo Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --allow config.json \
  --expect config.json=8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a \
  --expect-size config.json=726 \
  --allow generation_config.json \
  --expect generation_config.json=2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2 \
  --expect-size generation_config.json=239 \
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
  --allow merges.txt \
  --expect merges.txt=8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5 \
  --expect-size merges.txt=1671853 \
  --allow vocab.json \
  --expect vocab.json=ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 \
  --expect-size vocab.json=2776833 \
  --dest "$AUTORLBENCH_WORKSPACE/models/Qwen3-4B"

autorlbench_fetch_hf \
  --repo Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --allow config.json \
  --expect config.json=f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30 \
  --expect-size config.json=728 \
  --allow generation_config.json \
  --expect generation_config.json=2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2 \
  --expect-size generation_config.json=239 \
  --allow model-00001-of-00005.safetensors \
  --expect model-00001-of-00005.safetensors=31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f \
  --expect-size model-00001-of-00005.safetensors=3996250744 \
  --allow model-00002-of-00005.safetensors \
  --expect model-00002-of-00005.safetensors=5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282 \
  --expect-size model-00002-of-00005.safetensors=3993160032 \
  --allow model-00003-of-00005.safetensors \
  --expect model-00003-of-00005.safetensors=c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836 \
  --expect-size model-00003-of-00005.safetensors=3959604768 \
  --allow model-00004-of-00005.safetensors \
  --expect model-00004-of-00005.safetensors=b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a \
  --expect-size model-00004-of-00005.safetensors=3187841392 \
  --allow model-00005-of-00005.safetensors \
  --expect model-00005-of-00005.safetensors=20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff \
  --expect-size model-00005-of-00005.safetensors=1244659840 \
  --allow model.safetensors.index.json \
  --expect model.safetensors.index.json=f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc \
  --expect-size model.safetensors.index.json=32878 \
  --allow tokenizer.json \
  --expect tokenizer.json=aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 \
  --expect-size tokenizer.json=11422654 \
  --allow tokenizer_config.json \
  --expect tokenizer_config.json=d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101 \
  --expect-size tokenizer_config.json=9732 \
  --allow merges.txt \
  --expect merges.txt=8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5 \
  --expect-size merges.txt=1671853 \
  --allow vocab.json \
  --expect vocab.json=ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 \
  --expect-size vocab.json=2776833 \
  --dest "$AUTORLBENCH_WORKSPACE/models/Qwen3-8B"

# Harbor materializes the root-prepared snapshot from an unprivileged host process.
chmod -R a+rX "$AUTORLBENCH_WORKSPACE/models"

mkdir -p "$AUTORLBENCH_PREPARED"
printf 'PYTHONPATH=%s/areal:%s/tau2/src\nTAU2_DATA_DIR=%s/tau2/data\n' \
  "$AUTORLBENCH_WORKSPACE" "$AUTORLBENCH_WORKSPACE" "$AUTORLBENCH_WORKSPACE" \
  > "$AUTORLBENCH_PREPARED/env"
