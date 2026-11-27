#!/usr/bin/env bash
# slime_swe_sandbox_ttl_false_zero: what to prepare before training.
#
# Runs in a separate prepare-stage container: root, the
# `slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12` image,
# **the only stage with network**, and $AUTORLBENCH_WORKSPACE,
# $AUTORLBENCH_MODELS, and $AUTORLBENCH_DATASETS are all writable.
#
# Three assets: two downloaded (model + harness CLI), one converted on the
# spot. Everything else the task needs (ClawGym dataset rows and the vendored
# slime fork with the AGS sandbox backend) ships inside the workspace.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"

# --- Base model (--hf-checkpoint) --------------------------------------------
# scripts/run_grpo_local_swe.sh reads exactly $W/Qwen3-4B-Instruct-2507 via its
# HF_CKPT default; a different directory name makes megatron fail far from the
# cause.
# Pinned revision, for reference: Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f17c01a7cc42f958dc650907174af0554
python3 "$env_dir/autorlbench_fetch.py" hf \
    --repo Qwen/Qwen3-4B-Instruct-2507 \
    --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
    --allow LICENSE \
    --allow README.md \
    --allow config.json \
    --allow generation_config.json \
    --allow merges.txt \
    --allow model-00001-of-00003.safetensors \
    --allow model-00002-of-00003.safetensors \
    --allow model-00003-of-00003.safetensors \
    --allow model.safetensors.index.json \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.json \
    --expect model-00001-of-00003.safetensors=75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6 \
    --expect-size model-00001-of-00003.safetensors=3957900840 \
    --expect model-00002-of-00003.safetensors=0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1 \
    --expect-size model-00002-of-00003.safetensors=3987450520 \
    --expect model-00003-of-00003.safetensors=7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d \
    --expect-size model-00003-of-00003.safetensors=99630640 \
    --dest "$W/Qwen3-4B-Instruct-2507"

# --- Actor weights in mcore (torch_dist) format (--load) ---------------------
# There is no upstream repo; they are converted on the spot from the checkpoint
# above. scripts/run_grpo_local_swe.sh has its own guard and redoes this in the
# run stage when prepare could not (no GPU here), which is slower but correct.
#
# Pitfalls carried over from the slime conversion tool:
#   * The output is not byte-reproducible across world sizes; WORLD_SIZE=1
#     yields a single pp=1 layout. The training script loads it with TP2 and
#     reshards on load (verified against this exact task's recorded run).
#   * --rotary-base must be 5000000 (scripts/models/qwen3-4B-Instruct-2507.sh
#     sets it); conversion and training must agree on it.
SLIME_DIR="$W/slime"
HF_MODEL_DIR="$W/Qwen3-4B-Instruct-2507"
DIST_MODEL_DIR="$W/Qwen3-4B-Instruct-2507_torch_dist"

if [ -f "${DIST_MODEL_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "[prepare] torch_dist weights already present; skipping conversion"
elif python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "[prepare] converting $HF_MODEL_DIR -> $DIST_MODEL_DIR (WORLD_SIZE=1, rotary-base=5000000)"
    # shellcheck source=/dev/null
    source "$SLIME_DIR/scripts/models/qwen3-4B-Instruct-2507.sh"
    mkdir -p "$DIST_MODEL_DIR"
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    PYTHONPATH="$SLIME_DIR:/root/Megatron-LM" \
    python3 "$SLIME_DIR/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "$HF_MODEL_DIR" \
        --save "$DIST_MODEL_DIR"
else
    echo "[prepare] no CUDA device visible; leaving the torch_dist conversion to"
    echo "[prepare] the training script's own guard (scripts/run_grpo_local_swe.sh)."
fi

# --- Dataset sanity check ----------------------------------------------------
# The ClawGym dataset rows ship with the workspace (data_clawgym/swe_train.jsonl,
# 154 code-verified instances). This check confirms the file is present and
# non-empty; the sandbox-image references inside the rows point at the
# enterprise registry mirror and are resolved at rollout time by the fork's
# E2BSandbox AGS branch.
DATASET="$W/data_clawgym/swe_train.jsonl"
if [ -f "$DATASET" ]; then
    LINES=$(wc -l < "$DATASET")
    echo "[prepare] ClawGym dataset: $LINES instances"
else
    echo "[prepare] WARNING: dataset not found at $DATASET"
fi

# --- Harness CLI (claude-code platform binary) --------------------------------
# The AGS sandboxes are network-isolated, so the CLI tarball must be present
# on the trainer host for the per-episode chunked upload. The sandbox image
# does not ship it, and the run stage has no network. Downloaded here (the
# only networked stage) from the public npm registry, verified in-flight.
CC_VERSION="2.1.295"
CC_SHA256="d45a2fa14a7ea0b8ab4f56e502de5893c2b816078c71c4d8ab2e4142148f3d95"
CC_BYTES="114433914"
CC_URL="https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-${CC_VERSION}.tgz"
CC_TARBALL="$W/assets/claude-code-linux-x64-${CC_VERSION}.tgz"

if [ -f "$CC_TARBALL" ] && [ "$(sha256sum "$CC_TARBALL" | cut -d' ' -f1)" = "$CC_SHA256" ]; then
    echo "[prepare] harness CLI already present; skipping download"
else
    mkdir -p "$W/assets"
    echo "[prepare] downloading claude-code ${CC_VERSION} (${CC_BYTES} bytes)..."
    curl -sSL --retry 3 --speed-time 60 --speed-limit 100000 \
        -o "$CC_TARBALL" "$CC_URL"
    GOT_SHA=$(sha256sum "$CC_TARBALL" | cut -d' ' -f1)
    GOT_BYTES=$(stat -c%s "$CC_TARBALL")
    if [ "$GOT_SHA" != "$CC_SHA256" ] || [ "$GOT_BYTES" != "$CC_BYTES" ]; then
        echo "[prepare] ERROR: harness CLI checksum mismatch"
        echo "[prepare]   expected sha256=$CC_SHA256 bytes=$CC_BYTES"
        echo "[prepare]   got      sha256=$GOT_SHA bytes=$GOT_BYTES"
        rm -f "$CC_TARBALL"
        exit 1
    fi
    echo "[prepare] harness CLI downloaded and verified (${GOT_BYTES} bytes)"
fi
