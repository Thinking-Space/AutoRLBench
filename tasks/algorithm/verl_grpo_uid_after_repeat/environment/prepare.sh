#!/usr/bin/env bash
# verl_grpo_uid_after_repeat — prepare stage.
#
# Runs inside the `verl` Docker image (verlai/verl:vllm017.latest) with the verl
# environment already activated.  This is the only stage that may access the network.
#
# In the agent stage this file is mounted read-only at /autorlbench/environment
# and the agent can read it.  Maintainer notes (defect details, grading thresholds)
# go in ../README.md, not here.
set -euo pipefail

# --- Base model -------------------------------------------------------------
# examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh:30-38 searches, in order,
# /models/... -> /workspace/verl/Llama-3.2-1B-instruct -> a placeholder path.
# Nothing mounts the first candidate, so the model must land exactly on the second.
# Mind the case: the directory name is `Llama-3.2-1B-instruct` (lowercase i), which
# differs from the upstream repo name, but that is what the script expects.
autorlbench_fetch_hf \
    --repo LIF1014/autorlbench-Llama-3.2-1B-Instruct \
    --revision 32f9e721f796d3c639ab64d2f58ceeaa2c3213b9 \
    --allow config.json \
    --allow generation_config.json \
    --allow LICENSE.txt \
    --allow README.md \
    --allow USE_POLICY.md \
    --allow special_tokens_map.json \
    --allow tokenizer_config.json \
    --allow tokenizer.json \
    --allow tokenizer.model \
    --allow model.safetensors \
    --expect model.safetensors=1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f \
    --expect-size model.safetensors=2471645608 \
    --dest "$AUTORLBENCH_WORKSPACE/verl/Llama-3.2-1B-instruct"

# --- Dataset ----------------------------------------------------------------
# The same script (:44-62) requires /workspace/DAPO-Math-17k_verl/{train,test}.parquet.
# There is a single recipe (docker/autorlbench_datasets/dapo_math_17k.py), and tasks
# sharing this data reuse the same cache entry; their ground truth depends on identical
# bytes, so each task does not keep its own copy.
autorlbench_dataset \
    --recipe dapo_math_17k \
    --expect test.parquet=52596fd6ade0d49db3a49ef0c9c89f88aeb323c17fa5ee23fb0c774d7fba3255 \
    --expect-size test.parquet=331284 \
    --expect train.parquet=fc4cad54f2d15c622025b5e3764e91d2a79145e75358a703658daa66fe47d321 \
    --expect-size train.parquet=2735462 \
    --dest "$AUTORLBENCH_WORKSPACE/DAPO-Math-17k_verl"
