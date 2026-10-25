# AutoRLBench environment `ppo-critic-vimpo`.
# autorlbench-image: autorlbench/ppo-critic-vimpo:vllm0.11-torch2.8-cu128
#
# veRL PPO-Critic / VIMPO runtime, reconstructed from the environment that
# produced the recorded ppo_critic runs:
#   Python       3.12
#   CUDA         12.8
#   torch        2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0
#   vLLM         0.11.0
#   Ray          2.56.0
#   transformers 4.57.6
#   flash-attn   2.8.3.post1
#
# Tasks: ppo_critic/* (reward_parser_mismatch, excessive_actor_lr_collapse,
# proxy_reward_misalignment; aime25_single_sample_variance rides this image for
# its stdlib-only protocol check).
#
# This stack is DELIBERATELY NOT the shared `verl` image (vLLM 0.17 / torch 2.10
# / CUDA 12.9). The ppo_critic workspaces ship their own veRL + VIMPO recipe
# written against vLLM 0.11, and their recorded val-core / critic-reward
# baselines were produced on this exact stack; the solution grader reconstructs
# those baselines, so the stack must not move. The pins below are frozen for the
# same reason the `verl` image is pinned by digest: the tasks' recorded outcomes
# must not move with them. Changing a pin is a deliberate upgrade -- re-run the
# ppo_critic tasks on this environment afterward.
#
# The workspace veRL is NOT baked here: the harness bind-mounts each task's
# workspace at /workspace, and `PYTHONPATH=/workspace` (below) makes
# `import verl` resolve to that mounted source. That keeps a single image usable
# across every ppo_critic task despite each shipping its own veRL snapshot.
FROM docker.m.daocloud.io/nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python-is-python3 \
        git \
        git-lfs \
        curl \
        wget \
        ca-certificates \
        build-essential \
        ninja-build \
        cmake \
        pkg-config \
        libnuma-dev \
        libibverbs-dev \
        librdmacm-dev \
        libaio-dev && \
    rm -rf /var/lib/apt/lists/*

# Do not modify Ubuntu's apt-managed pip.
# All experiment Python packages live in this isolated virtual environment.
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/python --version && \
    /opt/venv/bin/python -m pip --version

RUN python -m pip config set global.index-url \
        https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
    python -m pip install --upgrade pip setuptools wheel

# Exact PyTorch CUDA 12.8 stack used by the original experiments.
RUN python -m pip install \
        torch==2.8.0 \
        torchvision==0.23.0 \
        torchaudio==2.8.0

# Exact rollout engine used by the original environment.
RUN python -m pip install \
        vllm==0.11.0

# Remaining packages captured from the working environment. Inlined here
# (rather than COPY'd) because docker/.dockerignore allowlists only
# autorlbench-activate.sh into the build context. Kept byte-identical to the
# bundle's environment/requirements-runtime.txt.
RUN python -m pip install -r /dev/stdin <<'REQS'
PyYAML==6.0.3
accelerate==1.14.0
aiohttp==3.14.1
antlr4-python3-runtime==4.9.3
codetiming==1.4.0
datasets==5.0.0
dill==0.4.1
einops==0.8.2
fastapi==0.138.2
fsspec==2025.9.0
huggingface_hub==0.36.2
hydra-core==1.3.3
liger_kernel==0.8.0
mathruler==0.1.0
mistral_common==1.11.5
numpy==2.2.6
nvidia-ml-py==13.610.43
omegaconf==2.3.1
openai==1.99.1
packaging==25.0
pandas==3.0.3
peft==0.19.1
protobuf==7.35.1
psutil==7.2.2
pyarrow==24.0.0
pybind11==3.0.4
pylatexenc==2.10
pynvml==13.0.1
qwen-vl-utils==0.0.14
ray==2.56.0
requests==2.34.2
safetensors==0.8.0
sentencepiece==0.2.1
setuptools==79.0.1
tensordict==0.10.0
tokenizers==0.22.2
torchdata==0.11.0
transformers==4.57.6
typing_extensions==4.15.0
uvicorn==0.49.0
wandb==0.28.0
xgrammar==0.1.25
tensorboard==2.21.0
tensorboard-data-server==0.7.2
REQS

# Exact flash-attn version used by the original experiments.
RUN MAX_JOBS=16 \
    python -m pip install \
        --no-build-isolation \
        flash_attn==2.8.3.post1

# Build-time consistency check. veRL itself is intentionally absent here (it
# arrives with the bind-mounted workspace), so it is not imported below.
RUN python - <<'PY'
import torch
import ray
import vllm
import transformers
import flash_attn

print("python env   = /opt/venv")
print("torch        =", torch.__version__)
print("torch cuda   =", torch.version.cuda)
print("ray          =", ray.__version__)
print("vllm         =", vllm.__version__)
print("transformers =", transformers.__version__)
print("flash_attn   =", flash_attn.__version__)

assert torch.__version__.startswith("2.8.0")
assert torch.version.cuda == "12.8"
assert ray.__version__ == "2.56.0"
assert vllm.__version__ == "0.11.0"
assert transformers.__version__ == "4.57.6"
assert "2.8.3" in flash_attn.__version__
PY

# ---- AutoRLBench tooling: keep identical in every docker/images/*.Dockerfile ----
# Build context is docker/ (see docker/.dockerignore for what it may COPY).

RUN mkdir -p /autorlbench/prepared /autorlbench/models /autorlbench/datasets

# Every command the harness runs goes through `sh -lc`, and a login shell's
# /etc/profile may reset PATH. Record the image's own PATH so the hook can put
# back whatever it drops (/opt/venv/bin, /usr/local/cuda/bin, ...).
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

CMD ["/bin/bash"]
