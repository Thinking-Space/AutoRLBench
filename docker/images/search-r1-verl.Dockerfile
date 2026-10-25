# AutoRLBench environment `search-r1-verl`.
# autorlbench-image: autorlbench/search-r1-verl:vllm0.6.3-torch2.4-cu121
#
# Search-R1 (with veRL vendored inside it) as of upstream commit
# 598e61bd1d36895726d28a8d06b3a15bed19f5d3: Python 3.10, torch 2.4.0, vLLM
# 0.6.3, flash-attn 2.6.3, transformers 4.46.3, ray 2.40.0, faiss-cpu 1.14.2,
# numpy 1.26.4.
# Tasks: search_r1_zero_query_vector.
#
# Why this is not the shared `verl` image: that one is vLLM 0.17 / torch 2.10 /
# CUDA 12.9, five minor versions of vLLM past the API this workspace's rollout
# code is written against. The recorded 32-step curves this task grades against
# were produced on the stack below, and `results/baseline32/experiment.json`
# names those versions.
#
# Unlike the other images here there is no upstream image to use as-is, so the
# stack is installed from the delivery's own `uv.lock` (copied to
# docker/search-r1-runtime/). That lock, not this Dockerfile, is the pin.
#
# UNVERIFIED: this image has never been built. It was written on a machine with
# no GPU and no reachable Docker Hub / GitHub, so:
#   * the FROM is pinned by tag, not by digest. Pin it at first build:
#       docker pull nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
#       docker inspect --format '{{index .RepoDigests 0}}' \
#         nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
#     and replace the tag with `@sha256:...`, the way every other image here is
#     pinned -- the tasks' recorded outcomes must not move with a moving tag.
#   * the flash-attn wheel URL below follows Dao-AILab's release naming for
#     2.6.3 but was not fetched. If it 404s the build falls back to compiling
#     the sdist from the lock, which is slow (~30-60 min at MAX_JOBS=4) but
#     produces the same version.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# git: `pip install -e` style sources inside the lock and anything that reads
# repo metadata at import. curl/ca-certificates: the wheel fetch below.
# build-essential: flash-attn's fallback path, and any sdist in the lock.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

# uv brings its own CPython 3.10 (the lock's requires-python is >=3.10,<3.12 and
# .python-version says 3.10), so the image does not depend on what the base
# distro ships.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

ENV UV_CACHE_DIR=/opt/uv-cache \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/opt/search-r1-runtime/.venv \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=4

WORKDIR /opt/search-r1-runtime
COPY search-r1-runtime/pyproject.toml search-r1-runtime/uv.lock search-r1-runtime/.python-version ./

# Two passes, the same split the delivery's setup_env.sh used: everything except
# flash-attn first (flash-attn's build imports torch, so torch has to exist
# before it is touched), then flash-attn.
#
# flash-attn comes from Dao-AILab's prebuilt wheel rather than the lock's sdist:
# cu123/torch2.4/cp310/cxx11abiFALSE matches this stack (PyPI torch 2.4.0 wheels
# are the pre-cxx11-ABI build), and it turns a 30-60 minute nvcc compile into a
# download. --no-deps keeps it from resolving torch again; einops is pinned to
# the lock's version because the wheel needs it and --no-deps will not bring it.
# If the URL is gone, fall back to the lock's own sdist so the build still
# produces flash-attn 2.6.3.
RUN set -eux; \
    uv sync --frozen --no-install-package flash-attn; \
    FA_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl; \
    if curl -fsSL --retry 3 -o /tmp/flash_attn.whl "$FA_WHEEL"; then \
        uv pip install --no-deps /tmp/flash_attn.whl einops==0.8.2; \
        rm -f /tmp/flash_attn.whl; \
    else \
        echo "prebuilt flash-attn wheel unavailable; compiling from the lock's sdist" >&2; \
        uv sync --frozen --no-build-isolation-package flash-attn; \
    fi; \
    rm -rf "$UV_CACHE_DIR"

# The venv is the image's Python: `python3` in any stage is the pinned
# interpreter, so workspace/run.py needs no --python and serve_retriever.py
# imports the same faiss/torch the training process does.
ENV PATH=/opt/search-r1-runtime/.venv/bin:$PATH \
    VIRTUAL_ENV=/opt/search-r1-runtime/.venv

RUN set -eux; \
    python3 -c "import torch, vllm, faiss, transformers, ray; \
print('torch', torch.__version__); print('vllm', vllm.__version__); \
print('faiss', faiss.__version__); print('transformers', transformers.__version__); \
print('ray', ray.__version__)"; \
    python3 -c "import flash_attn; print('flash-attn', flash_attn.__version__)"

# ---- AutoRLBench tooling: keep identical in every docker/images/*.Dockerfile ----
# Build context is docker/ (see docker/.dockerignore for what it may COPY).

RUN mkdir -p /autorlbench/prepared /autorlbench/models /autorlbench/datasets

# Every command the harness runs goes through `sh -lc`, and a login shell's
# /etc/profile may reset PATH. Record the image's own PATH so the hook can put
# back whatever it drops (/usr/local/cuda/bin, /root/.cargo/bin, ...).
RUN printf '%s\n' "$PATH" > /etc/autorlbench-image-path
COPY autorlbench-activate.sh /etc/profile.d/autorlbench-activate.sh
RUN set -eux; \
    sed -i 's/\r$//' /etc/profile.d/autorlbench-activate.sh; \
    chmod 0644 /etc/profile.d/autorlbench-activate.sh; \
    sh -n /etc/profile.d/autorlbench-activate.sh

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TOKENIZERS_PARALLELISM=false

# The prepare-stage fetch/dataset tooling (autorlbench_fetch.py, autorlbench_dataset.py,
# autorlbench_datasets/) is deliberately NOT baked here. docker/install-autorlbench-tooling.sh
# installs it into /usr/local at the start of every prepare run, from the repo's
# current copy (mounted read-only into the container). That keeps a fix to
# those files reaching running jobs without an image rebuild. Only the activate
# hook above stays baked -- it must exist in the agent/verifier containers too,
# which never run prepare-time provisioning.

# The harness bind-mounts a per-run copy of the task's workspace here.
WORKDIR /workspace
