# AutoRLBench recorded veRL 0.6.1 GRPO stack (build pending a container host).
# autorlbench-image: autorlbench/verl-v061-grpo:torch2.8-vllm0.11-cu128
FROM vllm/vllm-openai:v0.11.0@sha256:d8d39b59e909d2378ac4feeb191f7e7b6f1342477dc66b7c47cec89e9985ad8a
ENTRYPOINT []
CMD ["/bin/bash"]
COPY verl-v061-runtime/runtime.lock.txt /opt/autorlbench-runtime.txt
RUN python3 -m venv --system-site-packages /opt/autorlbench-venv
ENV PATH="/opt/autorlbench-venv/bin:${PATH}"
RUN python3 -c "import sys; assert sys.version_info[:2] == (3,12), sys.version" \
 && python3 -m pip install --no-cache-dir -r /opt/autorlbench-runtime.txt
RUN python3 -c "import torch,vllm; assert torch.__version__.split('+')[0]=='2.8.0'; assert vllm.__version__=='0.11.0'; assert torch._C._GLIBCXX_USE_CXX11_ABI"
RUN curl --fail --location --retry 3 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' \
 --output /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
 && echo 'f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb  /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' | sha256sum --check - \
 && python3 -m pip install --no-deps /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
 && rm /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
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
