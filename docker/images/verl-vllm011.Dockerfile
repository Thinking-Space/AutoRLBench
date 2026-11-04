# AutoRLBench environment `verl-vllm011`.
# autorlbench-image: autorlbench/verl-vllm011:vllm0.11-torch2.8
#
# PiSSA/LoRA GRPO tasks recorded with vLLM 0.11.0, torch 2.8.0,
# transformers 4.57.1, peft 0.18.0, datasets 4.0.0 and W&B 0.23.0.
# Tasks: infrastructure/verl_pissa_vllm_adapter_request.
#
# UNVERIFIED: this Dockerfile has not yet been built. The upstream image is
# pinned by digest; validate package installation and one four-step smoke run
# before publishing the image.
FROM verlai/verl:vllm011.dev5@sha256:6bb9db02c90d8aee762a377601c7d2e1abb7e7c1c1776827e5185f4d27f40185

RUN python3 -m pip install --no-cache-dir \
    'datasets==4.0.0' \
    'peft==0.18.0' \
    'transformers==4.57.1' \
    'wandb==0.23.0'

# ---- AutoRLBench tooling: keep identical in every docker/images/*.Dockerfile ----
# Build context is docker/ (see docker/.dockerignore for what it may COPY).

RUN mkdir -p /autorlbench/prepared /autorlbench/models /autorlbench/datasets

RUN printf '%s\n' "$PATH" > /etc/autorlbench-image-path
COPY autorlbench-activate.sh /etc/profile.d/autorlbench-activate.sh
RUN set -eux; \
    sed -i 's/\r$//' /etc/profile.d/autorlbench-activate.sh; \
    chmod 0644 /etc/profile.d/autorlbench-activate.sh; \
    sh -n /etc/profile.d/autorlbench-activate.sh

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace
