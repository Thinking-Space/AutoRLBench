#!/usr/bin/env bash
# verl_grpo_ppo_mini_batch_too_small: prepare stage.
#
# Runs in a separate prepare-stage container: root, the verl-vllm0.24.0
# image, **the only networked stage**, and $AUTORLBENCH_WORKSPACE,
# $AUTORLBENCH_MODELS, and $AUTORLBENCH_DATASETS are all writable.
#
# Two assets: the Qwen2.5-0.5B-Instruct checkpoint (988 MB) and the raw openai/gsm8k files (2.7 MB).
# The RL parquet files (data/gsm8k_boxed/{train,test}.parquet) ship with
# the workspace; they are what the shipped examples/data_preprocess/gsm8k_boxed.py (upstream
# gsm8k.py with the \boxed{} instruction) produces from the raw files below, and
# run_gsm8k_grpo.sh rebuilds them from data/raw/gsm8k only if one is missing.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"

# --- scorer pins, shadowed over the image -------------------------------------
# The image (vLLM 0.24 / torch 2.11 / flash-attn 2.8.3) carries the runtime
# stack the workspace-vendored verl 0.9.0 needs, but not the Math-Verify
# scorer pins the reward depends on: math-verify 0.9.0 and its parser
# latex2sympy2_extended (the pins of the former verl-0.9-vllm0.28 recipe).
# antlr4-python3-runtime is pinned to 4.9.3, inside latex2sympy2_extended's
# accepted range and the exact version the image's omegaconf 2.3.1 requires —
# antlr4 4.13 (math-verify's default resolution) breaks omegaconf's
# deserialized grammar. sympy/mpmath are left to the image (1.14.0).
# --no-deps: the three pins are the complete closure; 4.9.3 ships sdist
# only, so it installs from source with --no-build-isolation.
site="$AUTORLBENCH_PREPARED/pysite"
if [ ! -d "$site/math_verify" ]; then
    _pip_args=(--no-cache-dir --no-deps --target "$site")
    if [ -n "${AUTORLBENCH_PIP_INDEX_URL:-}" ]; then
        _pip_args+=(--index-url "$AUTORLBENCH_PIP_INDEX_URL")
        if [ -n "${AUTORLBENCH_PIP_TRUSTED_HOST:-}" ]; then
            _pip_args+=(--trusted-host "$AUTORLBENCH_PIP_TRUSTED_HOST")
        fi
    fi
    python3 -m pip install "${_pip_args[@]}" \
        'math-verify==0.9.0' 'latex2sympy2_extended==1.11.0'
    python3 -m pip install --no-cache-dir --no-deps --target "$site" \
        --no-build-isolation 'antlr4-python3-runtime==4.9.3'
fi
# Shadow for the rest of this script and for the agent/rerun/verifier stages
# via prepared/env. run_gsm8k_grpo.sh prepends the workspace (vendored verl)
# on top of this, so the layering stays workspace > pysite > image.
export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
if ! grep -qs "^PYTHONPATH=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
    echo "PYTHONPATH=$site" >> "${AUTORLBENCH_PREPARED}/env"
fi

# --- image-level env, seeded here --------------------------------------------
# The two entries the recording rig's env carried ride the prepared/env
# handoff into the agent/rerun/verifier stages via the activate hook.
# WANDB_MODE/WANDB_API_KEY are exported by the harness itself for every
# stage. Idempotent.
for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# The task-local fetch_hf.py fetches the whole repos, cache-first, without
# checksum verification; the revisions (and per-file pins) the former
# autorlbench_fetch_hf allowlist recorded, for reference:
#   Qwen/Qwen2.5-0.5B-Instruct @ 7ae557604adf67be50417f59c2c2f167def9a775
#   openai/gsm8k @ 740312add88f781978c0658806c59bc2815b9866
python3 "$env_dir/fetch_hf.py" \
    --repo Qwen/Qwen2.5-0.5B-Instruct \
    --dest "$W/models/Qwen2.5-0.5B-Instruct"
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo openai/gsm8k \
    --dest "$W/data/raw/gsm8k"
