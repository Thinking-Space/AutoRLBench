#!/usr/bin/env bash
# r1v_clevr_format_only_reward: prepare stage.
#
# Runs in a separate prepare-stage container: root, the
# `verl-vllm0.7.4-torch2.5.1-flashattn2.7.4-cu124-py3.10` image, **the only
# networked stage**, and $AUTORLBENCH_WORKSPACE is writable.
#
# In the agent stage this file is mounted read-only and the agent can read it.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_PREPARED:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"

# --- Python layer -------------------------------------------------------------
# The trainer in open_r1/ is written against TRL 0.14 and the transformers
# commit its upstream pins; the image ships transformers 4.49.0 and has no trl,
# deepspeed, qwen-vl-utils or math-verify. antlr4 4.13.2 replaces the image's
# 4.9.3, which latex2sympy2_extended (a math-verify dependency) cannot run on.
site="$AUTORLBENCH_PREPARED/pysite"
if [ ! -f "$site/.complete" ]; then
    rm -rf "$site"
    mkdir -p "$site"
    stage="$(mktemp -d "$AUTORLBENCH_PREPARED/pipstage.XXXXXX")"
    _pip_args=(--no-cache-dir --no-deps --target "$site")
    if [ -n "${AUTORLBENCH_PIP_INDEX_URL:-}" ]; then
        _pip_args+=(--index-url "$AUTORLBENCH_PIP_INDEX_URL")
        if [ -n "${AUTORLBENCH_PIP_TRUSTED_HOST:-}" ]; then
            _pip_args+=(--trusted-host "$AUTORLBENCH_PIP_TRUSTED_HOST")
        fi
    fi
    TMPDIR="$stage" python3 -m pip install "${_pip_args[@]}" \
        trl==0.14.0 deepspeed==0.15.4 hjson==3.1.0 qwen-vl-utils==0.0.10 \
        math-verify==0.5.2 latex2sympy2_extended==1.0.6 antlr4-python3-runtime==4.13.2

    # transformers at the commit the trainer's upstream pins (4.50.0.dev0).
    # Fetched by URL, so size and SHA-256 are verified in flight.
    python3 "$env_dir/fetch_url.py" \
        --url "https://github.com/huggingface/transformers/archive/92c5ca9dd70de3ade2af2eb835c96215cc50e815.zip" \
        --sha256 57156c5d67e5049142ea84a0b5a0739346565630d7cf20d208a2d0673c2801a2 \
        --size 25783384 \
        --dest "$stage/transformers.zip"
    TMPDIR="$stage" python3 -m pip install "${_pip_args[@]}" "$stage/transformers.zip"
    rm -rf "$stage"
    touch "$site/.complete"
fi
export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
if ! grep -qs "^PYTHONPATH=" "$AUTORLBENCH_PREPARED/env" 2>/dev/null; then
    echo "PYTHONPATH=$site" >> "$AUTORLBENCH_PREPARED/env"
fi
python3 -c 'from importlib.metadata import version
assert version("transformers") == "4.50.0.dev0", version("transformers")
assert version("trl") == "0.14.0", version("trl")
assert version("deepspeed") == "0.15.4", version("deepspeed")'

# --- Base model ---------------------------------------------------------------
# run_grpo.sh:10 reads MODEL_DIR, default $WS/models/Qwen2-VL-2B-Instruct.
# Pinned revision, for reference:
#   Qwen/Qwen2-VL-2B-Instruct @ 895c3a49bc3fa70a340399125c650a463535e71c
python3 "$env_dir/fetch_hf.py" \
    --repo Qwen/Qwen2-VL-2B-Instruct \
    --dest "$W/models/Qwen2-VL-2B-Instruct"

# --- Datasets -----------------------------------------------------------------
# run_grpo.sh:11 reads DATA_DIR, default $WS/data, and expects three
# `datasets.save_to_disk` directories below it: train/, val/, ood_val/.
# build_data.py derives them from two public repos, so the raw repos go to a
# staging directory that is removed once the splits exist.
# Pinned revisions, for reference:
#   leonardPKU/clevr_cogen_a_train @ d5a8da66aa97c9c19ebe4075b7c7f777fef850a6
#   RyanWW/Super-CLEVR             @ 2ecf1d00d1fa06b4a4ed9330269ac57d2f411090
if [ ! -f "$W/data/.complete" ]; then
    raw="$W/.autorlbench-raw"
    python3 "$env_dir/fetch_hf.py" --type dataset \
        --repo leonardPKU/clevr_cogen_a_train --dest "$raw/clevr_cogent_a"
    python3 "$env_dir/fetch_hf.py" --type dataset \
        --repo RyanWW/Super-CLEVR --dest "$raw/super_clevr"
    python3 "$env_dir/build_data.py" --raw "$raw" --out "$W/data"
    rm -rf "$raw"
fi
