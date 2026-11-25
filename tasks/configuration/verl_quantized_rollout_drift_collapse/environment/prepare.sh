#!/usr/bin/env bash
# verl experiment prepare stage: fetch the start model, patch the vendored verl.
#
# Runs in a separate prepare-stage container: root, the
# vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12 image, **the only networked
# stage**, and $AUTORLBENCH_WORKSPACE is writable. It fetches the start model and
# applies one fix to the vendored verl tree (see the patch section below). verl
# v0.8.0 (commit 7aed6b230776f963fa09509c10d9c3a767d1102c) ships in workspace/verl
# otherwise verbatim; the image already carries verl's Python dependencies. The
# Countdown source parquet ships in workspace/raw/countdown/data/ and run_grpo.sh
# builds the RL parquet from it at launch.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"
mkdir -p "$W/.tmp" "$W/.home"
export HOME="$W/.home" TMPDIR="$W/.tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

if ! grep -qs "^TOKENIZERS_PARALLELISM=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
    echo "TOKENIZERS_PARALLELISM=false" >> "${AUTORLBENCH_PREPARED}/env"
fi

# --- Base model ---------------------------------------------------------------
# run_grpo.sh defaults MODEL to $W/models/Qwen2.5-1.5B-Instruct.
# Pinned revision (immutable commit; the revision is part of the fetch cache key):
# Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
python3 "$env_dir/fetch_hf.py" \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    --dest "$W/models/Qwen2.5-1.5B-Instruct"

# --- Vendored verl fix: online fp8 weight sync on vLLM 0.11 ---------------------
# Upstream verl @ 7aed6b2 crashes the first fp8 weight sync on vLLM 0.11: the
# per-shard blockwise scales do not tile the fused qkv/gate_up parameters, and
# the SM90-transposed scales of square-scale layers (o_proj) come in
# mis-oriented. Any run that enables the rollout weight quantization needs this
# fix to get past the first sync. The patch is applied unconditionally, its two
# sha256 states are tracked in environment/source_manifest.json, and the
# recorded evidence was produced with this exact file.
FP8="$W/verl/verl/utils/vllm/vllm_fp8_utils.py"
FP8_PATCHED_SHA="d86cfd6589992e41c698f15ccf3692f2c0be18f8f7589f74ab7a9c32602494b3"
FP8_UPSTREAM_SHA="530f8f9c90840cab671393968c54841daaeb5a97039cae65bd68414c977bba36"
got_sha() { sha256sum "$1" | cut -d' ' -f1; }
if [ "$(got_sha "$FP8")" = "$FP8_PATCHED_SHA" ]; then
    echo "[prepare] vllm_fp8_utils.py already patched"
elif [ "$(got_sha "$FP8")" = "$FP8_UPSTREAM_SHA" ]; then
    cp "$env_dir/patches/vllm_fp8_utils.py" "$FP8"
    [ "$(got_sha "$FP8")" = "$FP8_PATCHED_SHA" ] || {
        echo "[prepare] vllm_fp8_utils.py patch did not land" >&2; exit 1; }
    echo "[prepare] vllm_fp8_utils.py patched (online fp8 weight sync fix)"
else
    echo "[prepare] unexpected vllm_fp8_utils.py state, refusing to patch" >&2
    exit 1
fi
