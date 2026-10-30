# autorlbench-image: autorlbench/trl-grpo-torch26:py312-cu124
# Versions from the paired TRL runs recorded on 2026-10-30. Not yet built.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir setuptools==80.9.0 wheel==0.45.1 \
    && python3 -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -m pip install --no-cache-dir transformers==5.15.1 datasets==4.7.0 accelerate==1.15.0 \
       wandb==0.22.3 numpy==2.2.6 safetensors==0.8.0 tokenizers==0.22.2 peft==0.18.1 rich==14.2.0

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

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
