# autorlbench-image: autorlbench/areal-tau2:torch210-vllm018
FROM modelbest-kt8zv1.swr-pro.myhuaweicloud.com/ptg/ptg-verl@sha256:7f849b97a9bbc0efec9540e066bf1ac0ff48314d06eb7d1cd4acb2cf9e4129bc
RUN python -m pip install --no-cache-dir vllm==0.18.0 transformers==4.57.6 'tensordict==0.10.0'
RUN python -m pip install --no-cache-dir https://codeload.github.com/Ascend/TransferQueue/zip/5c15a2c8d65f05a52514830da6de4c5ec4336c1c
# Runtime used for the tau2 paired rollout case, 2026-11-04.
RUN python3 -m pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple --ignore-installed blinker==1.9.0
RUN python3 -m pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    litellm==1.81.3 openai==2.24.0 pydantic==2.12.5 colorlog==6.12.0 awex==0.8.1 \
    redis==8.1.0 deepdiff==9.1.0 addict==2.4.0 fs==2.4.16 docstring-parser==0.17.0 \
    pydantic-argparse==0.10.0 gymnasium==1.4.0 hydra-core==1.4.0.dev1 omegaconf==2.4.0.dev2 \
    toml==0.10.2 zstandard==0.25.0 langfuse==2.60.7 watchdog==7.0.0 colorama==0.4.6 \
    tensorboardx==2.6.5 pebble==5.2.3 prettytable==3.18.0 nltk==3.10.4 \
    word2number==1.1 aiofiles==25.1.0 networkx==3.3 tenacity==9.1.4 Flask==3.1.3

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
