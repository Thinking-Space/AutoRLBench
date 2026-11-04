# AutoRLBench environment `verl-0.9-vllm0.28`.
# autorlbench-image: autorlbench/verl-0.9-vllm0.28:torch2.13-cu130
#
# verl 0.9.0 stack: vLLM 0.28.0, torch 2.13.0 / CUDA 13.0, transformers 5.10.4,
# ray 2.58.0, tensordict 0.10.0, flash-attn 2.8.3 (built from source: there is no torch-2.13
# wheel upstream). verl itself is not installed: the task workspace carries verl v0.9.0 and its
# launch script puts it first on PYTHONPATH.
# Tasks: reward/verl_grpo_gsm8k_any_number_reward.
#
# The versions are those of the conda env both recorded runs of this task used (1 x RTX A6000,
# SM86, 2026-10-27). The base is the vLLM v0.28.0 release image, which already ships the
# torch 2.13.0 + CUDA 13.0 pair vLLM 0.28.0 pins; the digest is that tag's multi-arch index.
#
# UNVERIFIED: this Dockerfile has not been built yet. The recorded runs used the conda env, not
# this image.
FROM vllm/vllm-openai:v0.28.0@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14

RUN set -eux; \
    python3 -m pip install --no-cache-dir \
        transformers==5.10.4 tokenizers==0.22.2 huggingface_hub==1.33.0 safetensors==0.8.0 \
        "ray[default]==2.58.0" tensordict==0.10.0 torchdata==0.11.0 \
        hydra-core==1.3.7 omegaconf==2.3.1 codetiming==1.4.0 dill==0.4.1 \
        accelerate==1.15.0 peft==0.21.0 datasets==5.0.1 pandas==3.0.6 pyarrow==25.0.1 \
        numpy==2.5.4 pydantic==2.13.5 regex==2026.9.10 pybind11==3.1.0 \
        pylatexenc==2.11 math-verify==0.9.0 wandb==0.30.0 nvidia-ml-py==13.615.71 \
        packaging==25.0 ninja; \
    python3 -c "import torch, vllm; assert torch.__version__.startswith('2.13.0'), torch.__version__; print(torch.__version__, vllm.__version__)"

# verl's FSDP actor defaults to attn_implementation=flash_attention_2 with use_remove_padding.
# Arch list covers A6000 (8.6), H100 (9.0) and RTX PRO 6000 (12.0).
RUN set -eux; \
    FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="8.6;9.0;12.0" MAX_JOBS=16 NVCC_THREADS=2 \
        python3 -m pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3; \
    python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)"

# The upstream image's entrypoint is `vllm serve`; the harness runs its own commands.
ENTRYPOINT []

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
