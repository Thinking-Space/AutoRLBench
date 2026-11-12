#!/usr/bin/env bash
# slime_search_r1_nomask: what to prepare before training.
#
# Identical to the healthy twin task's prepare.sh: both tasks run the **same
# recipe** (examples/search-r1/run_qwen2.5_3B.sh) and need exactly the same
# assets. Any difference in the container environment or the assets makes the
# two runs incomparable, so a change here must be mirrored there.
#
# Runs in a separate prepare-stage container: root, the `slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12` image,
# **the only stage with network**, and $AUTORLBENCH_WORKSPACE, $AUTORLBENCH_MODELS,
# and $AUTORLBENCH_DATASETS are all writable. The framework enforces the
# agent/verifier stage offline policy itself (HF_HUB_OFFLINE /
# TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE in runtimes/base.py's
# _network_policy_env), so **do not** repeat those in the task's
# metadata environment env: that env applies to all three stages alike and
# would switch off the network prepare itself needs.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"

# ---------------------------------------------------------------------------
# faiss-gpu -- only the search-r1 tasks need it
# ---------------------------------------------------------------------------
# Why it is not in docker/images/slime-v0.3.2.Dockerfile: that image is shared
# by every slime task, and adding a package would change the recorded
# environment of the retool tasks. Their outcomes were measured on the stack
# without faiss, and another task's requirement must not move it.
#
# Why the GPU build instead of faiss-cpu (the original environment/Dockerfile
# had faiss-cpu behind an `import yaml || pip install ...` guard that never
# fired): local_dense_retriever/retrieval_server.py:211-214 calls
# faiss.GpuMultipleClonerOptions + faiss.index_cpu_to_all_gpus, neither of
# which exists in the CPU build; the recipe also starts the service with
# --faiss_gpu (run_qwen2.5_3B.sh:43). Version 1.15.0 is the one recorded in
# the historical run's wandb freeze.
#
# Why faiss-gpu-cuvs and not faiss-gpu: faiss 1.15 ships two GPU wheels for
# two CUDA layouts, and this environment is CUDA 13. faiss-gpu is the CUDA 12
# per-component layout: at import it looks for .so files under the
# `nvidia.cuda_runtime` / `nvidia.cublas` / `nvidia.curand` nvidia-*-cu12
# modules, none of which this environment has (the image pins
# torch==2.11.0+cu129 plus the unsuffixed nvidia-cuda-runtime==13.0.9x line,
# which lays out as a single nvidia/cu13/lib/ directory). Forcing a set of
# cu12 wheels in does not work either: retrieval_server.py imports faiss and
# torch in the same process, and the two libcurand.so.10 copies (the cu12 one
# and the cu13 one share the soname) collide inside the same CUDA context.
# faiss-gpu-cuvs is the one built for CUDA 13 and reads nvidia/cu13/lib/.
#
# Why the cuVS line is pinned at 26.6 rather than the latest 26.8:
# libcuvs-cu13 26.8 requires nvidia-nvjitlink>=13.3 while the image pins
# 13.0.88, and importing would die on `undefined symbol:
# __nvJitLinkCreate_13_3`. 26.6 only requires >=13.0, and lands inside the
# >=26.06,<27 range faiss-gpu-cuvs 1.15.0 itself asks for.
#
# Why libraft / librmm / rapids-logger are listed one by one: --no-deps is
# deliberate (next block), and libcuvs.so's DT_NEEDED chain is
# rapids-logger -> librmm -> libraft -> libcuvs, the order faiss/__init__.py
# also preloads. One missing link is one "cannot open shared object file".
#
# Why it installs into $AUTORLBENCH_PREPARED instead of the image
# site-packages: prepare is a **one-shot container** that is removed around
# the script (the harness's prepare stage), so anything installed into the
# image's site-packages is gone when the container exits. $AUTORLBENCH_PREPARED
# is the only directory the later two stages mount read-only. About 640MB.
FAISS_SITE="$AUTORLBENCH_PREPARED/pysite"
# The source for the network fallback path (reached only when the wheelhouse
# below is unavailable): the cluster's own proxying index proxies
# pythonhosted wheels over and is far more stable than the direct route --
# the whitelist-proxy resets partway through these few-hundred-MB RAPIDS cu13
# wheels (SSLEOFError: UNEXPECTED_EOF_WHILE_READING), the same failure class
# as the blocked HF xet cas-bridge. The default is the Paratera beijing
# cluster's index; override with AUTORLBENCH_PIP_INDEX_URL /
# AUTORLBENCH_PIP_TRUSTED_HOST when switching clusters instead of
# hard-coding one cluster's address into the task, and set it to the empty
# string to force the original direct-pythonhosted behaviour.
PIP_INDEX_URL="${AUTORLBENCH_PIP_INDEX_URL-https://pypi.mb1.bj3.paratera.com/root/pypi/+simple}"
PIP_TRUSTED_HOST="${AUTORLBENCH_PIP_TRUSTED_HOST-pypi.mb1.bj3.paratera.com}"
PIP_INDEX_ARGS=()
if [ -n "$PIP_INDEX_URL" ]; then
    PIP_INDEX_ARGS+=(--index-url "$PIP_INDEX_URL")
    [ -n "$PIP_TRUSTED_HOST" ] && PIP_INDEX_ARGS+=(--trusted-host "$PIP_TRUSTED_HOST")
fi

# --- Where the wheels come from: a wheelhouse shipped with the task, installed
# offline ----------------------------------------------------------------------
# The ten wheels (the same set the two pip blocks below install) do not go
# through any PyPI index: they were pre-downloaded from files.pythonhosted.org
# on a machine with open egress and published as one HF repo
# (cyh2004/ptdbench-wheelhouse-cu13, with a MANIFEST.txt recording each
# wheel's source URL and sha256); prepare pulls it locally with the same
# fetch_hf.py as the weights -- the same proven endpoint/mirror route, cached
# under $AUTORLBENCH_MODELS so the 1.33GB is fetched once and reused across
# runs and tasks -- and installs with `pip --no-index --find-links`.
#
# Why it has to go this far (measured 2026-11, devspace 1441367 /
# cluster=huhehaote):
#   * The beijing mirror pypi.mb1.bj3.paratera.com (the default of the
#     network fallback path above) is unreachable from a huhehaote devspace
#     outright -- TCP timeout in a shell, SSLEOFError resets ten in a row
#     inside the prepare container through the whitelist proxy. It was picked
#     for cluster=beijing.
#   * pypi.org's simple index pages connect, but files.pythonhosted.org is
#     reset by the whitelist proxy during the TLS handshake
#     (SSL_ERROR_SYSCALL, 5/5 reproduced) -- degraded from the old "resets
#     partway through" to "cannot handshake at all".
#   * The inner-mongolia mirror pypi.hs1.paratera.com is reachable and fast,
#     but has not synced the RAPIDS cu13 family at all
#     (faiss-gpu-cuvs / libcuvs / libraft / librmm / rapids-logger);
#     nvidia-cusolver stops at 12.3.4.7 and nvidia-cublas at 13.8.0.4 --
#     only three of the ten pinned versions are covered.
# In other words, no pip source on the huhehaote cluster can supply this
# closure; switching mirrors does not help.
#
# AUTORLBENCH_WHEELHOUSE_REPO overrides the repo id. A failed fetch or an
# incomplete wheelhouse (a sha256 mismatch counts as incomplete -- pip itself
# does not verify wheel contents) falls back to the network path above, which
# still works on a beijing devspace.
WHEELHOUSE_REPO="${AUTORLBENCH_WHEELHOUSE_REPO-cyh2004/ptdbench-wheelhouse-cu13}"
WHEELHOUSE="$AUTORLBENCH_PREPARED/wheelhouse"
# Filenames + sha256, matching the repo's MANIFEST.txt verbatim; the digests
# come from the link fingerprints on PyPI's simple index pages (verified one
# by one at download time).
WHEELHOUSE_MANIFEST='
nvidia_nvjitlink-13.4.92-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl  e0391f24ed94ec879b84e3da4d4ec320c879aff681f2c7a638462f7199284323
nvidia_cublas-13.8.2.12-py3-none-manylinux_2_27_x86_64.whl  21721774d8e7044debec3034ecc1a8cd22e173eb6dae5d7f5f8a014852382894
nvidia_curand-10.4.4.72-py3-none-manylinux_2_27_x86_64.whl  25c3457ae7a224fdd484dab90b0fc5dc0e842fab5db3012afa4a5bd2af4eb7e5
nvidia_cusolver-12.3.5.3-py3-none-manylinux_2_27_x86_64.whl  01f13254cfe9f99cf8d141fa4d00fbe55b235fd1509854017d766fd84442f46a
nvidia_cuda_nvrtc-13.4.92-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl  5ce8c97b00b232c4f50c8c4b5a3b68cafee08bdb82ea86f2052ff01d03194f4a
faiss_gpu_cuvs-1.15.0-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl  a1bc663066606ee5c9cd8e72db100c3ba0a5a24b62f7149b24b03749f156d127
libcuvs_cu13-26.6.0-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl  32f4b705a8f261c310af0e7624d32e68190b071b148021c20bae793ba1ddec22
libraft_cu13-26.6.0-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl  945d3e40adb3d47f5dade5fe7819d1331b6ed9e22799cbcb215281946f94b858
librmm_cu13-26.6.0-py3-none-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl  58dd98261f74ed0df523a93c063d182b8f1a8ee6b8afb46566e08f8829f28e13
rapids_logger-0.2.3-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl  7fe67ef4049c5d8ba6154746325dcf7cc0f327f0efa8f2611fc8f64e67510f60
'
wheelhouse_ready() {
    local line name digest
    while IFS= read -r line; do
        case "$line" in ''|'#'*) continue ;; esac
        name="${line%% *}"; digest="${line##* }"
        [ -s "$WHEELHOUSE/$name" ] || return 1
        if command -v sha256sum >/dev/null 2>&1; then
            [ "$(sha256sum "$WHEELHOUSE/$name" | awk '{print $1}')" = "$digest" ] || return 1
        fi
    done <<<"$WHEELHOUSE_MANIFEST"
    return 0
}
# The fetch goes in two steps: a cache hit never leaves the network at all,
# and a cache miss first goes through the environment's whitelist proxy, then
# retries once completely bypassing it. Why the direct step exists: HF puts
# every file over 10MB on the xet backend, whose resolve 302s to
# cas-bridge.xethub.hf.co, a domain the whitelist proxy does not allow (TLS
# handshake reset, measured 2026-11-03 on a wuxi_train devspace); the devspace
# node's direct connection to the mirror works instead (transparent egress),
# with the same file sha256 as on the PyPI side. Only when both routes fail
# does the wheelhouse give up and fall back to pip's network path.
fetch_wheelhouse() {
    python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
        --repo "$WHEELHOUSE_REPO" \
        --dest "$WHEELHOUSE" && return 0
    echo "[prepare] wheelhouse fetch through the egress proxy failed; retrying direct (no proxy)" >&2
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
        --repo "$WHEELHOUSE_REPO" \
        --dest "$WHEELHOUSE"
}
PIP_SOURCE_ARGS=()
PIP_SOURCE_DESC=""
if fetch_wheelhouse && wheelhouse_ready; then
    PIP_SOURCE_ARGS=(--no-index --find-links "$WHEELHOUSE")
    PIP_SOURCE_DESC="wheelhouse $WHEELHOUSE_REPO (offline)"
else
    echo "[prepare] wheelhouse unavailable; falling back to the pip index path" >&2
    PIP_SOURCE_ARGS=(${PIP_INDEX_ARGS[@]+"${PIP_INDEX_ARGS[@]}"})
    PIP_SOURCE_DESC="${PIP_INDEX_URL:-direct pypi.org}"
fi

# The five nvidia cu13 components (nvjitlink / cublas / curand / cusolver /
# nvrtc) go into system dist-packages, not $FAISS_SITE: the faiss loader
# resolves `nvidia.cu13` at import, and the 0-byte nvidia/__init__.py stub in
# dist-packages makes nvidia a regular package, so a $FAISS_SITE/nvidia
# installed with --target never takes part in `import nvidia.cu13` resolution
# -- the loader only follows dist-packages' __path__[0]. The image does have a
# nvidia/ directory, but it is the cu12 layout that ships with torch 2.11
# (nvidia/nvjitlink/lib/libnvJitLink.so.12 and friends); the .so set the
# faiss-gpu-cuvs loader wants under nvidia/cu13/lib/ is only present there as
# cudart/nvvm/nvblas. Without the first three (the .13-soname nvjitlink /
# cublas / curand), import dies at "failed to load libnvJitLink.so.13" --
# three of the five libraries the faiss loader preloads itself with
# RTLD_GLOBAL.
#
# The last two are missing deeper: libcuvs.so's DT_NEEDED lists
# libcusolver.so.12 and libnvrtc.so.13 (the image only has .so.11 / .so.12
# there; the torch cu12 layout has neither soname). After the five preloads,
# rapids' load.py dlopens libcuvs.so and, when it is missing, returns None
# **silently**; _swigfaiss.abi3.so then reports "libcuvs.so: cannot open
# shared object file" -- three layers away from the cause. The cusolver
# wheel's version is 12.3.5.3 rather than 13.x: along the CUDA 13 line
# cusolver's **library version** is 12 (soname .so.12), and the unsuffixed
# nvidia-cusolver series stops at 12.3.5.3, still installing into
# nvidia/cu13/lib/ (the other four components use the same layout; checked
# against each RECORD). nvrtc's distribution name carries the cuda- prefix
# (nvidia-cuda-nvrtc, 13.4.92).
#
# These five are distinct distribution names from the image's existing
# *-cu12 packages (nvidia-nvjitlink vs the nvidia-nvjitlink-cu12 series), so
# the pip metadata does not conflict and the sonames do not collide (.13 vs
# .12): in the retrieval server process, torch (cu12 wheels) and faiss (cu13
# directory) each load their own set; importing both in one process plus a
# GPU cloner smoke test passed. cuda-runtime is not installed: the image's
# nvidia-cuda-runtime==13.3.29 already puts libcudart.so.13 in cu13/lib/.
# This block sits after the PIP_SOURCE_ARGS decision so the source is chosen
# there (wheelhouse first, network path as the fallback; the unsuffixed cu13
# wheels are on pypi.org these days -- the cluster just cannot reach
# files.pythonhosted.org). It writes
# the prepare container's system directories, and the guard reinstalls them
# after a devspace rebuild (this task runs on an attach-mode devspace only,
# with prepare/agent/verifier sharing the container). Versions: nvjitlink
# 13.4.92 / cublas 13.8.2.12 / curand 10.4.4.72 / cusolver 12.3.5.3 /
# cuda-nvrtc 13.4.92, which all provide the __nvJitLinkCreate_13_0 symbol
# libcuvs 26.6 needs.
NV13_SYS="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib"
cu13_ready() {
    local soname
    for soname in libnvJitLink.so.13 libcublas.so.13 libcurand.so.10 \
                  libcusolver.so.12 libnvrtc.so.13; do
        [ -e "$NV13_SYS/$soname" ] || return 1
    done
    return 0
}
if ! cu13_ready; then
    echo "[prepare] installing CUDA-13 nvidia runtime components into system dist-packages"
    python3 -m pip install --no-cache-dir --no-deps --no-user \
        --retries 10 --timeout 120 \
        ${PIP_SOURCE_ARGS[@]+"${PIP_SOURCE_ARGS[@]}"} \
        nvidia-nvjitlink==13.4.92 \
        nvidia-cublas==13.8.2.12 \
        nvidia-curand==10.4.4.72 \
        nvidia-cusolver==12.3.5.3 \
        nvidia-cuda-nvrtc==13.4.92
else
    echo "[prepare] CUDA-13 nvidia runtime components already present"
fi
# The guard checks this version's dist-info, not the `faiss/` directory: with
# a directory check, a half-installed tree or a stale cu12 one would look
# complete and only blow up in the agent stage.
FAISS_MARKER="$FAISS_SITE/faiss_gpu_cuvs-1.15.0.dist-info"
if [ ! -d "$FAISS_MARKER" ]; then
    if [ -e "$FAISS_SITE" ]; then
        echo "[prepare] $FAISS_SITE is stale or incomplete; reinstalling from scratch"
        rm -rf "$FAISS_SITE"
    fi
    echo "[prepare] installing faiss-gpu-cuvs==1.15.0 (CUDA 13) into $FAISS_SITE via $PIP_SOURCE_DESC"
    # --no-user: some images set PIP_USER=1 (or pip.conf [install] user=true),
    # which is mutually exclusive with --target ("Can not combine '--user' and
    # '--target'").
    python3 -m pip install --no-cache-dir --no-deps --no-user \
        --retries 10 --timeout 120 \
        ${PIP_SOURCE_ARGS[@]+"${PIP_SOURCE_ARGS[@]}"} \
        --target "$FAISS_SITE" \
        faiss-gpu-cuvs==1.15.0 \
        libcuvs-cu13==26.6.0 \
        libraft-cu13==26.6.0 \
        librmm-cu13==26.6.0 \
        rapids-logger==0.2.3
else
    echo "[prepare] faiss already present in $FAISS_SITE"
fi

# libcuvs.so's RPATH is relative to **pysite**:
# $ORIGIN:$ORIGIN/../../nvidia/cu13/lib:$ORIGIN/../../nvidia/nccl/lib:$ORIGIN/../../libcuvs_cu13.libs.
# The system install (previous block) does not reach its cu13 / nccl elements;
# rapids' load.py (libcuvs/load.py's _load_wheel_installation) returns None
# **silently** when dlopen of libcuvs.so fails, and _swigfaiss.abi3.so then
# dies at "libcuvs.so: cannot open shared object file". This lays a symlink
# layer in pysite pointing back at the system directories so those two RPATH
# elements hit: $FAISS_SITE/nvidia/cu13/lib -> dist-packages/nvidia/cu13/lib
# (the five components plus cudart live there), nccl likewise (shipped by the
# image's torch cu12 layout; ldconfig already resolves it, the layer just
# stops it depending on ldconfig). $FAISS_SITE/nvidia has no __init__.py, so
# under PEP 420 it never joins `import nvidia.cu13` resolution (the
# dist-packages regular package always wins) -- this layer is purely for the
# dynamic linker. $FAISS_SITE is a fresh directory every run, so every
# prepare pass rebuilds it (ln -sfn is idempotent).
NVROOT="$(python3 -c 'import nvidia.cu13, os; print(os.path.dirname(nvidia.cu13.__path__[0]))')"
mkdir -p "$FAISS_SITE/nvidia/cu13" "$FAISS_SITE/nvidia/nccl"
ln -sfn "$NVROOT/cu13/lib" "$FAISS_SITE/nvidia/cu13/lib"
ln -sfn "$NVROOT/nccl/lib" "$FAISS_SITE/nvidia/nccl/lib"

# The later two stages see faiss through this line. Note that the slime launch
# script **assigns** PYTHONPATH rather than appending
# (`export PYTHONPATH=/workspace:/root/Megatron-LM`, and writes it again inside
# the ray runtime_env), so this line never reaches the training process --
# which is fine: the training side does not import faiss. The one that does is
# the standalone retrieval server, and it runs in a plain shell that sees it.
if ! grep -qs "^PYTHONPATH=" "$AUTORLBENCH_PREPARED/env"; then
    echo "PYTHONPATH=$FAISS_SITE" >> "$AUTORLBENCH_PREPARED/env"
fi

# --- Weights -----------------------------------------------------------------
# The base policy.
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen2.5-3B \
    --dest "$AUTORLBENCH_WORKSPACE/Qwen2.5-3B"

# The e5 encoder for retrieval.
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo intfloat/e5-base-v2 \
    --dest "$AUTORLBENCH_WORKSPACE/e5-base-v2"

# --- Datasets ----------------------------------------------------------------
# Train / eval parquets and the retrieval corpus + FAISS index, both via the
# standard in-container `autorlbench_dataset` (installed from the repo's
# docker/autorlbench_datasets before this script runs). The two recipes pin
# revisions, assert row counts / byte counts, and land in the shared
# $AUTORLBENCH_DATASETS cache, hardlinked per run.
autorlbench_dataset \
    --recipe search_r1_nq_hotpotqa \
    --dest "$AUTORLBENCH_WORKSPACE/nq_hotpotqa_train"

autorlbench_dataset \
    --recipe wiki18_e5 \
    --dest "$AUTORLBENCH_WORKSPACE/wiki-18"

# --- Pre-warm the corpus arrow cache ----------------------------------------
# The retrieval service's load_corpus calls
# `datasets.load_dataset("json", data_files=..., split="train", num_proc=4)`
# (local_dense_retriever/retrieval_server.py:34-36), so every cold start
# converts the 14GB jsonl into an arrow tree. The conversion belongs to
# prepare: it only reads the already-staged corpus, needs no GPU, and the
# result is reusable across runs -- left to the agent stage it is ten-plus
# minutes of waiting inside the timed rerun.
#
# The process does not survive; the cache does -- this is the one thing
# prepare can help with for the retrieval service. The service itself is not
# started here; see the end of the file.
#
# Two directories instead of one is forced by read/write permissions:
#
#   * $AUTORLBENCH_DATASETS survives across runs, but the agent / verifier
#     stages mount it read-only, and datasets opens a .lock in the cache
#     directory even on a cache hit, which fails on read-only.
#   * $AUTORLBENCH_WORKSPACE is writable in both stages, but it is a
#     per-run copy and empty at the start of every run.
#
# So the real cache is built under the datasets root and hardlinked into the
# workspace: on the same filesystem that copy costs no bytes (the same trick
# the other materialize steps in this file use), and the lock file the agent
# stage writes is a new file landing on the workspace side, never touching the
# cross-run copy.
#
# A hit requires the fingerprint to match: the datasets config id includes the
# data_files path and mtime. The path is /workspace/wiki-18/wiki-18.jsonl in
# both stages (the in-container path agrees), and the mtime follows the
# hardlinked inode, so it is stable across runs. If it ever misses, the cost
# is only the service converting once more -- no error -- which is also why
# the whole block below only warns on failure instead of exiting.
HF_CACHE_PERSISTENT="$AUTORLBENCH_DATASETS/hf_datasets_cache"
HF_CACHE_RUN="$AUTORLBENCH_WORKSPACE/.hf_datasets_cache"
CORPUS_JSONL="$AUTORLBENCH_WORKSPACE/wiki-18/wiki-18.jsonl"

warm_corpus_cache() {
    mkdir -p "$HF_CACHE_PERSISTENT"
    # num_proc matches retrieval_server.py's line exactly: it is part of the
    # converted output's path, so a different value converts a second tree.
    HF_DATASETS_CACHE="$HF_CACHE_PERSISTENT" \
    CORPUS_JSONL="$CORPUS_JSONL" \
    python3 - <<'PY'
import os

import datasets

corpus = os.environ["CORPUS_JSONL"]
print(f"[prepare] building the arrow cache for {corpus}", flush=True)
rows = datasets.load_dataset("json", data_files=corpus, split="train", num_proc=4)
print(f"[prepare] corpus cached: {len(rows)} rows", flush=True)
PY
    # -a preserves mtime (the fingerprint uses it), -l makes the copy cost an
    # inode instead of bytes. -l fails across filesystems, and then an honest
    # copy is the fallback: slow, but better than no cache.
    rm -rf "$HF_CACHE_RUN"
    cp -al "$HF_CACHE_PERSISTENT" "$HF_CACHE_RUN" 2>/dev/null \
        || cp -a "$HF_CACHE_PERSISTENT" "$HF_CACHE_RUN"
}

if warm_corpus_cache; then
    echo "[prepare] corpus arrow cache staged at $HF_CACHE_RUN"
else
    # The cache is an optimization, not an asset. On failure the retrieval
    # server converts at startup, ten-plus minutes slower but still correct;
    # exiting here would fail the whole prepare together with the ~90GB
    # already downloaded.
    echo "[prepare] warming the corpus arrow cache failed; the retrieval server" >&2
    echo "[prepare] will do the conversion itself on startup (slower, still correct)." >&2
    rm -rf "$HF_CACHE_RUN"
fi

# The later stages see the cache through this line -- the workspace copy is
# the writable one. Same mechanism as the faiss PYTHONPATH above: prepared/env
# is a KEY=VALUE file the harness exports in every stage
# (docker/autorlbench-activate.sh).
if ! grep -qs "^HF_DATASETS_CACHE=" "$AUTORLBENCH_PREPARED/env"; then
    echo "HF_DATASETS_CACHE=$HF_CACHE_RUN" >> "$AUTORLBENCH_PREPARED/env"
fi

# --- Reference weights in mcore (torch_dist) format --------------------------
SLIME_DIR="$AUTORLBENCH_WORKSPACE"
HF_MODEL_DIR="$AUTORLBENCH_WORKSPACE/Qwen2.5-3B"
DIST_MODEL_DIR="$AUTORLBENCH_WORKSPACE/Qwen2.5-3B_torch_dist"

CONVERT_GPU="$(
python3 <<'PY' 2>/dev/null
import sys

try:
    import torch
except Exception:
    sys.exit(0)
if not torch.cuda.is_available():
    sys.exit(0)

best, most_free = None, 0
for index in range(torch.cuda.device_count()):
    try:
        free, _total = torch.cuda.mem_get_info(index)
    except Exception:
        # A GPU too full to even create a context is not usable.
        continue
    if free > most_free:
        best, most_free = index, free

# 12GiB: enough for the 6GB weights plus model-construction peak, with half
# the headroom to spare. Skip rather than run out of memory halfway.
if best is not None and most_free >= (12 << 30):
    print(best)
PY
)" || CONVERT_GPU=""

if [ -n "$CONVERT_GPU" ]; then
    echo "[prepare] converting $HF_MODEL_DIR -> $DIST_MODEL_DIR (GPU $CONVERT_GPU, WORLD_SIZE=1)"
    # shellcheck source=/dev/null
    source "$SLIME_DIR/scripts/models/qwen2.5-3B.sh"
    mkdir -p "$DIST_MODEL_DIR"
    if CUDA_VISIBLE_DEVICES="$CONVERT_GPU" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
       PYTHONPATH="/root/Megatron-LM:$SLIME_DIR" \
       python3 "$SLIME_DIR/tools/convert_hf_to_torch_dist.py" \
           "${MODEL_ARGS[@]}" \
           --hf-checkpoint "$HF_MODEL_DIR" \
           --save "$DIST_MODEL_DIR"; then
        echo "[prepare] torch_dist reference ready at $DIST_MODEL_DIR"
    else
        # A half-finished dist checkpoint is far worse than none: the training
        # script's guard only checks the directory exists and is non-empty, so
        # a half-converted tree looks done to it and --ref-load then reads
        # wrong weights.
        echo "[prepare] conversion failed; dropping $DIST_MODEL_DIR so the training" >&2
        echo "[prepare] script's own guard redoes it (run_qwen2.5_3B.sh:79-86)." >&2
        rm -rf "$DIST_MODEL_DIR"
    fi
else
    echo "[prepare] no GPU with at least 12GiB free (or no GPU at all); leaving the"
    echo "[prepare] torch_dist conversion to the training script's own guard"
    echo "[prepare] (examples/search-r1/run_qwen2.5_3B.sh:79-86)."
fi

echo "[prepare] assets staged under $AUTORLBENCH_WORKSPACE"
echo "[prepare] the retrieval server is not started here; run_qwen2.5_3B.sh brings it"
echo "[prepare] up itself (examples/search-r1/start_retriver.sh) and waits for :8000."
