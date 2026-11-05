# AutoRLBench environment `verl-vllm011`.
# autorlbench-image: autorlbench/verl-vllm011:vllm0.11-torch2.8
#
# BUILD RECIPE, not a pull file (see docker/README.md "Exceptions"): a thin
# layer over the digest-pinned upstream verl image. Build and push it once,
# then collapse this file to the single digest-pinned
# `FROM autorlbench/verl-vllm011:vllm0.11-torch2.8@sha256:...` line the
# convention expects. Until that push exists, pulling the tag above fails —
# do not run the tasks before building.
#
# PiSSA/LoRA GRPO tasks recorded with vLLM 0.11.0, torch 2.8.0,
# transformers 4.57.1, peft 0.18.0, datasets 4.0.0 and W&B 0.23.0.
# Tasks: algorithm/verl_pissa_residual_base_reference,
# infrastructure/verl_pissa_vllm_adapter_request.
#
# Restored from the pre-2026-11-05 recipe (git show
# 87c6be6c^:docker/images/verl-vllm011.Dockerfile), adapted to the
# bake-nothing convention: no /autorlbench mount-point mkdir, no
# /etc/profile.d/autorlbench-activate.sh (activation is injected at runtime
# by autorlbench/runtimes/base.py), and no image-level ENV —
# CUDA_DEVICE_ORDER / TOKENIZERS_PARALLELISM are seeded by the tasks'
# prepare.sh into $AUTORLBENCH_PREPARED/env instead.
# UNVERIFIED as in the original recipe: validate package installation and one
# four-step smoke run before publishing the image.
FROM verlai/verl:vllm011.dev5@sha256:6bb9db02c90d8aee762a377601c7d2e1abb7e7c1c1776827e5185f4d27f40185

RUN python3 -m pip install --no-cache-dir \
    'datasets==4.0.0' \
    'peft==0.18.0' \
    'transformers==4.57.1' \
    'wandb==0.23.0'
