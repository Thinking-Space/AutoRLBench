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

# --- image-level env, seeded here --------------------------------------------
# Post-port docker/images/*.Dockerfile are bare digest-pinned FROM lines: no
# ENV is baked into the image any more. The two entries the old purpose-built
# image carried are seeded into $AUTORLBENCH_PREPARED/env here instead; they
# ride the prepared/env handoff into the agent/rerun/verifier stages via the
# activate hook. Idempotent.
for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

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

# --- Megatron-LM version pin --------------------------------------------------
# The calibration image ships Megatron-LM already at 4716f754 (full history);
# the vendored miles tree in this workspace still imports the legacy
# megatron.training.tokenizer module (removed upstream 2026-02-09), and the
# throughput thresholds were calibrated on this commit (the author's stack).
# The checkout is a no-op guard that also covers a drifted future image.
# Unlike the devspace-based PTDBench run, each AutoRLBench stage runs in a
# fresh container, so this pin only holds for the torch_dist conversion below
# -- the agent/rerun stages re-check the era via the run_spec command prefix.
git -C /root/Megatron-LM checkout --quiet --force 4716f75475c78e2fc2c6f0d3af095f1681b770b4

# --- sglang era guard ----------------------------------------------------------
# In the calibration image sglang is a REGULAR (non-editable) install under the
# system python's dist-packages (0.5.16.dev44 lineage, baked at image build):
# the checkout at /sgl-workspace/sglang is a shallow reference clone, and
# checking out a different commit there changes nothing at runtime, so there is
# no pin to apply -- only a guard that the runtime is still the era the
# thresholds were calibrated on. The vendored miles tree needs
# parameter_mapper.py and a dataclass ServerArgs (the post-v0.5.20-rebase
# sglang breaks both: dataclasses.fields(ServerArgs) TypeErrors in
# sglang_engine.py:783), and the binary kernels must match (sgl_kernel 0.4.4;
# 0.4.9 renamed/dropped ~30 top-level symbols the era code imports).
SGLANG_SITE="$(python3 -c 'import sglang, os; print(os.path.dirname(sglang.__file__))')"
if [ ! -f "${SGLANG_SITE}/srt/model_loader/parameter_mapper.py" ]; then
    echo "[prepare] sglang at ${SGLANG_SITE} lacks srt/model_loader/parameter_mapper.py"
    echo "[prepare] (post-v0.5.20-rebase lineage): the vendored miles tree cannot import it."
    exit 1
fi
python3 - <<'PYEOF'
import dataclasses
from sglang.srt.server_args import ServerArgs

if not dataclasses.is_dataclass(ServerArgs):
    raise SystemExit(
        "[prepare] ServerArgs is not a dataclass (post-v0.5.20-rebase sglang): "
        "miles sglang_engine.py dataclasses.fields(ServerArgs) would TypeError."
    )
PYEOF

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
