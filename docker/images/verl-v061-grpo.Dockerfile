# AutoRLBench recorded veRL 0.6.1 GRPO stack (build pending a container host).
# autorlbench-image: autorlbench/verl-v061-grpo:torch2.8-vllm0.11-cu128
FROM vllm/vllm-openai:v0.11.0@sha256:d8d39b59e909d2378ac4feeb191f7e7b6f1342477dc66b7c47cec89e9985ad8a
ENTRYPOINT []
CMD ["/bin/bash"]
COPY verl-v061-runtime/runtime.lock.txt /opt/verl-v061-runtime.lock
RUN python3 -m venv --system-site-packages /opt/verl-v061-venv
ENV PATH="/opt/verl-v061-venv/bin:${PATH}"
RUN python3 -c "import sys; assert sys.version_info[:2] == (3,12), sys.version" \
 && python3 -m pip install --no-cache-dir -r /opt/verl-v061-runtime.lock
RUN python3 -c "import torch,vllm; assert torch.__version__.split('+')[0]=='2.8.0'; assert vllm.__version__=='0.11.0'; assert torch._C._GLIBCXX_USE_CXX11_ABI"
RUN curl --fail --location --retry 3 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' \
 --output /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
 && echo 'f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb  /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' | sha256sum --check - \
 && python3 -m pip install --no-deps /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
 && rm /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# ---- adapted to the bake-nothing convention (2026-11-06) ----
# The venv path and lock file were renamed from /opt/autorlbench-* to
# /opt/verl-v061-* (internal paths; the image has never been built). No
# /autorlbench mount-point mkdir, no /etc/profile.d/autorlbench-activate.sh
# (activation is injected at runtime by autorlbench/runtimes/base.py), and no
# image-level ENV: CUDA_DEVICE_ORDER / TOKENIZERS_PARALLELISM are seeded by
# the tasks' prepare.sh into $AUTORLBENCH_PREPARED/env instead.
#
# BUILD RECIPE, not a pull file (see docker/README.md "Exceptions"): the venv
# layer over the digest-pinned upstream vLLM image must be built and pushed
# once as autorlbench/verl-v061-grpo:torch2.8-vllm0.11-cu128, then this file
# collapses to a single digest-pinned FROM line. Until that push exists, the
# tasks' image pull fails -- build first. Tasks using it:
# algorithm/verl_policy_loss_padded_mask_only,
# algorithm/verl_policy_loss_padded_unmasked,
# infrastructure/verl_rollout_weight_overwrite (not yet reworked).
