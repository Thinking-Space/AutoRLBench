#!/usr/bin/env bash
# Prepare assets before training.
#
# Runs in a separate prepare-stage container: root, the slime-v0.3.2-cu129
# image, the only networked stage, and $AUTORLBENCH_WORKSPACE,
# $AUTORLBENCH_MODELS and $AUTORLBENCH_DATASETS are all writable.
#
# MIGRATED (2026-11-11) from the never-built ngc-2506 calibration image to the
# slime cu129 image (same lmsysorg/sglang CI lineage, cu12 flavor): the image's
# binary layer already carries the same-generation stack -- sgl-kernel
# 0.4.4+cu129, flashinfer 0.6.12, flash-attn 2.8.3 + fa3-hopper (source-built
# against the image's torch 2.11.0+cu129), apex, TE 2.16.1, ray[default],
# mooncake cu12, sgl-router. What this prepare adds on top, via pysite/PYTHONPATH
# shadow (nothing is written into the image):
#   - the sglang-miles fork d218d6c7 (0.5.16.dev44 lineage) -- the image's
#     sglang is stock 0.5.15.post1 + slime patches, NOT the fork the vendored
#     miles tree's engine surface (pause_generation mode / weight update /
#     group recycling) is built against;
#   - Megatron-LM 4716f754 (the image ships slime's 1dcf0daf + patches);
#   - mbridge (ISEEKYAN/mbridge@89eb1088, the author's docker/Dockerfile pin):
#     miles_plugins/mbridge/* imports the top-level `mbridge` package
#     eagerly, and tools/convert_hf_to_torch_dist.py:12 does
#     `import miles_plugins.mbridge`, so the conversion (and the era guard
#     path through it) hard-fails without it. Found during the 2026-11-11
#     devspace boot verification;
#   - megatron-energon is deliberately NOT installed (see below).
# The recorded throughput thresholds (92/148 steps) are cu130-calibration
# numbers and must be re-recorded on this stack before grading.
#
# The DAPO-Math-17k dedup prompt set (data/dapo_math_17k_dedup.jsonl) ships in
# the workspace, so there is no dataset fetch here. One asset is downloaded (the
# base model) and one is converted on the spot (its torch_dist reference).
set -euo pipefail

# --- image-level env, seeded here --------------------------------------------
# Seeded into $AUTORLBENCH_PREPARED/env; they ride the prepared/env handoff
# into the agent/rerun/verifier stages via the activate hook. Idempotent.
for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# --- pysite: the sglang-miles fork + Megatron-LM 4716f754 -------------------
# Installed into $AUTORLBENCH_PREPARED (pure Python; the binary closure
# -- torch / sgl-kernel / flashinfer / flash-attn / apex / TE -- stays
# image-native). PYTHONPATH precedence puts the fork ahead of the image's
# editable sglang 0.5.15.post1 and the Megatron clone ahead of the image's
# slime-patched 1dcf0daf. --no-deps: the image carries the transitive closure.
# (megatron-energon is deliberately NOT installed: the vendored miles tree
# never imports it -- the calibration image only carried it as an inert
# dependency of the author's docker/Dockerfile.)
site="$AUTORLBENCH_PREPARED/pysite"
SGLANG_COMMIT=d218d6c7835307da50373f81704e61338b4e4847
if [ ! -d "$site/sglang" ]; then
    python3 -m pip install --no-cache-dir --no-deps --target "$site" \
        "sglang @ git+https://github.com/sgl-project/sglang@${SGLANG_COMMIT}#subdirectory=python"
fi
megatron_src="$AUTORLBENCH_PREPARED/Megatron-LM"
if [ ! -d "$megatron_src/.git" ]; then
    git clone https://github.com/radixark/Megatron-LM.git "$megatron_src"
fi
git -C "$megatron_src" checkout --quiet --force 4716f75475c78e2fc2c6f0d3af095f1681b770b4

# mbridge (ISEEKYAN/mbridge@89eb1088, the author's docker/Dockerfile pin):
# miles_plugins/mbridge/* imports the top-level `mbridge` package eagerly, and
# tools/convert_hf_to_torch_dist.py:12 does `import miles_plugins.mbridge`, so
# both the conversion and the runtime plugin chain hard-fail without it
# (found during the 2026-11-11 devspace boot verification).
if [ ! -d "$site/mbridge" ]; then
    python3 -m pip install --no-cache-dir --no-deps --target "$site" \
        "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c"
fi

# polars: miles/dashboard/store.py:44 imports it unconditionally (rollout
# manager's import chain), and the slime image does not carry it. NOT installed
# with --no-deps: the polars wheel is a pure-Python shell whose binary lives in
# the separate polars_runtime_32 dependency wheel (1.42.1 split packaging).
# The rest of the workspace requirements.txt the slime image already carries
# (accelerate/datasets/wandb/transformers 5.12.1/xxhash 3.8.1-vs-3.7.1 etc.,
# see the audit); the still-missing entries (hypothesis, psycopg,
# nvidia-resiliency-ext, torchft-nightly, qwen_vl_utils) are never imported
# by the vendored miles tree on this task's code path.
if [ ! -d "$site/polars" ]; then
    python3 -m pip install --no-cache-dir --target "$site" "polars==1.42.1"
fi

# Export for the rest of this script, and persist the composed PYTHONPATH for
# the agent/rerun/verifier stages via prepared/env (train_grpo.sh prepends the
# workspace on top of this). Idempotent: repeated runs don't stack entries.
export PYTHONPATH="$megatron_src:$site${PYTHONPATH:+:$PYTHONPATH}"
if ! grep -qs "^PYTHONPATH=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
    echo "PYTHONPATH=$megatron_src:$site" >> "${AUTORLBENCH_PREPARED}/env"
fi

# --- sglang era guard ----------------------------------------------------------
# The runtime sglang must be the fork at d218d6c7 (0.5.16.dev44 lineage,
# pre-v0.5.20-rebase): the vendored miles tree needs parameter_mapper.py and a
# dataclass ServerArgs (the post-rebase sglang breaks both:
# dataclasses.fields(ServerArgs) TypeErrors in sglang_engine.py:783), and the
# binary kernels must match (sgl-kernel 0.4.4; 0.4.9 renamed/dropped ~30
# top-level symbols the era code imports). The image's sgl-kernel is the
# official 0.4.4+cu129 build, and the fork install above shadows the image's
# stock 0.5.15.post1, so both halves are asserted here.
SGLANG_SITE="$(python3 -c 'import sglang, os; print(os.path.dirname(sglang.__file__))')"
if [ "${SGLANG_SITE}" != "$site/sglang" ]; then
    echo "[prepare] sglang resolves to ${SGLANG_SITE}, expected the fork under $site"
    exit 1
fi
if [ ! -f "${SGLANG_SITE}/srt/model_loader/parameter_mapper.py" ]; then
    echo "[prepare] sglang at ${SGLANG_SITE} lacks srt/model_loader/parameter_mapper.py"
    echo "[prepare] (post-v0.5.20-rebase lineage): the vendored miles tree cannot import it."
    exit 1
fi
python3 - <<'PYEOF'
import dataclasses
import megatron.training.tokenizer  # noqa: F401 -- legacy module, removed upstream 2026-02-09
import transformer_engine
from sglang.srt.server_args import ServerArgs

if not dataclasses.is_dataclass(ServerArgs):
    raise SystemExit(
        "[prepare] ServerArgs is not a dataclass (post-v0.5.20-rebase sglang): "
        "miles sglang_engine.py dataclasses.fields(ServerArgs) would TypeError."
    )
print("[prepare] TE", transformer_engine.__version__, "| megatron OK | sglang fork OK")
PYEOF

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
    PYTHONPATH="$megatron_src:$site:$WS" \
    python3 "$WS/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "$HF_MODEL_DIR" \
        --save "$DIST_MODEL_DIR"
else
    echo "[prepare] no CUDA device visible; leaving the torch_dist conversion to"
    echo "[prepare] the training script's own guard (scripts/train_grpo.sh)."
fi
