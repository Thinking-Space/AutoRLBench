#!/usr/bin/env bash
# Prepare stage.
#
# Runs in a separate prepare-stage container: root, the
# `vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12` image, **the only
# networked stage**, and $AUTORLBENCH_WORKSPACE is writable.
#
# In the agent stage this file is mounted read-only and the agent can read it.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_PREPARED:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"

# --- Python layer -------------------------------------------------------------
# The image carries torch 2.8.0, vLLM 0.11.0, transformers 4.57.6, peft 0.19.1,
# datasets 5.0.0 and flash-attn 2.8.3, which is everything the workspace needs.
# TRL is not installed: train_grpo.py imports the vendored copy in
# workspace/trl (vendored from TRL 0.27.2), as the recorded
# run did. Check that nothing else on the path shadows it.
(cd "$W" && python3 -c 'import trl, pathlib, sys
assert pathlib.Path(trl.__file__).resolve().parent == pathlib.Path("trl").resolve(), trl.__file__')

# --- Base model ---------------------------------------------------------------
# run_grpo.sh:5 reads MODEL, default models/Qwen2.5-1.5B-Instruct.
# Pinned revision, for reference:
#   Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 (Apache-2.0)
python3 "$env_dir/fetch_hf.py" \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --dest "$W/models/Qwen2.5-1.5B-Instruct"

# --- Dataset ------------------------------------------------------------------
# train_grpo.py reads data/gsm8k/main/train-00000-of-00001.parquet and
# eval_gsm8k.py reads data/gsm8k/main/test-00000-of-00001.parquet.
# Pinned revision, for reference:
#   openai/gsm8k @ 740312add88f781978c0658806c59bc2815b9866 (MIT)
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo openai/gsm8k \
    --dest "$W/data/gsm8k"
