#!/usr/bin/env bash
# Prepare assets before training.
#
# Runs in a separate prepare-stage container: root, the `miles` image,
# **the only networked stage**, and $AUTORLBENCH_WORKSPACE, $AUTORLBENCH_MODELS and
# $AUTORLBENCH_DATASETS are all writable.
#
# The DAPO-Math-17k dedup prompt set (data/dapo_math_17k_dedup.jsonl) ships in
# the workspace, so there is no dataset fetch here. One asset is downloaded (the
# base model) and one is converted on the spot (its torch_dist reference).
set -euo pipefail

# --- miles image venv --------------------------------------------------------
# The image's training stack (torch / sglang / ray) lives in a uv venv under
# /opt/sglang, not the system python. Both this prepare and the SSH-driven
# rerun run non-login shells, so bare `python3` falls to the empty
# /usr/bin/python3 and `import torch` dies. Put the venv on PATH for the
# torch_dist conversion below, and persist it into $AUTORLBENCH_PREPARED/env:
# the activate hook exports it in the agent/rerun/verifier stages
# (train_grpo.sh needs `python3` and `ray`). Idempotent: repeated installs
# don't stack entries.
if [ -d /opt/sglang/bin ] && ! grep -qs "^PATH=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
    echo "PATH=/opt/sglang/bin:${PATH}" >> "${AUTORLBENCH_PREPARED}/env"
fi
if [ -x /opt/sglang/bin/python3 ]; then
    export PATH="/opt/sglang/bin:${PATH}"
fi

# --- Base model (--hf-checkpoint) --------------------------------------------
# scripts/train_grpo.sh reads --hf-checkpoint "$WS/Qwen3-1.7B", so --dest must be
# exactly $AUTORLBENCH_WORKSPACE/Qwen3-1.7B. The fetch is whole-repo and
# cache-first (see fetch_hf.py); it does NOT checksum-verify what it
# downloads -- integrity rests on the repo id being a public, stable name and
# on the cache dir having been populated by an earlier verified fetch.
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen3-1.7B \
    --dest "$AUTORLBENCH_WORKSPACE/Qwen3-1.7B"

# --- Megatron torch_dist reference weights (--ref-load) ----------------------
# No upstream repo; converted from the HF checkpoint above. scripts/train_grpo.sh
# does this itself (its `if [ ! -d "$WS/Qwen3-1.7B_torch_dist" ]` guard) when it
# runs; doing it here keeps the GPU-hours for training. The miles source tree is
# in the task workspace, not the image.
#
# Conversion needs CUDA + NCCL (dist.init_process_group + torch.cuda.set_device).
# The harbor backend runs prepare as a GPU-less build layer, so **skip instead
# of erroring** there and let the training script's guard do it in the agent
# stage, where /workspace is still writable. The dist-checkpoint shard layout
# follows WORLD_SIZE; pinning it to 1 gives the pp=1 single-shard layout the
# training run (tp=pp=1) reloads regardless of machine.
WS="$AUTORLBENCH_WORKSPACE"
HF_MODEL_DIR="$WS/Qwen3-1.7B"
DIST_MODEL_DIR="$WS/Qwen3-1.7B_torch_dist"

if python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "[prepare] converting $HF_MODEL_DIR -> $DIST_MODEL_DIR (WORLD_SIZE=1)"
    # shellcheck source=/dev/null
    source "$WS/scripts/models/qwen3-1.7B.sh"
    mkdir -p "$DIST_MODEL_DIR"
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    PYTHONPATH="/root/Megatron-LM:$WS" \
    python3 "$WS/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "$HF_MODEL_DIR" \
        --save "$DIST_MODEL_DIR"
else
    echo "[prepare] no CUDA device visible; leaving the torch_dist conversion to"
    echo "[prepare] the training script's own guard (scripts/train_grpo.sh)."
fi
