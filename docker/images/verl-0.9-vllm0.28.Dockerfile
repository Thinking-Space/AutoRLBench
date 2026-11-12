# AutoRLBench environment `verl-0.9-vllm0.28`.
# autorlbench-image: autorlbench/verl-0.9-vllm0.28:torch2.13-cu130
#
# verl 0.9.0 stack: vLLM 0.28.0, torch 2.13.0 / CUDA 13.0, transformers 5.10.4,
# ray 2.58.0, tensordict 0.10.0, flash-attn 2.8.3 (built from source: there is no torch-2.13
# wheel upstream). verl itself is not installed: the task workspace carries verl v0.9.0 and its
# launch script puts it first on PYTHONPATH.
# Tasks: configuration/verl_grpo_ppo_mini_batch_too_small.
#
# The versions are those of the conda env both recorded runs of these tasks used (1 x RTX A6000,
# SM86, 2026-10-27). The base is the vLLM v0.28.0 release image, which already ships the
# torch 2.13.0 + CUDA 13.0 pair vLLM 0.28.0 pins; the digest is that tag's multi-arch index.
#
# Restored build recipe (2026-11-06) from the pre-87c6be6c docker/ tree, adapted to the
# bake-nothing convention: no /autorlbench mounts, no activate hook, no image ENV (the
# CUDA_DEVICE_ORDER / TOKENIZERS_PARALLELISM the recording rig's env carried are seeded
# into prepared/env by each task's prepare.sh instead).
#
# UNVERIFIED: this Dockerfile has not been built yet. The recorded runs used the conda env, not
# this image. Build and push as autorlbench/verl-0.9-vllm0.28:torch2.13-cu130 once, then
# collapse it to a single digest-pinned FROM line.
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
