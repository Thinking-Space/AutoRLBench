# AutoRLBench environment `trl-grpo-torch26`.
# autorlbench-image: autorlbench/trl-grpo-torch26:py312-cu124
#
# TRL GRPO stack from the paired runs recorded on 2026-10-30: Python 3.12,
# torch 2.6.0 / CUDA 12.4, transformers 5.15.1, datasets 4.7.0, accelerate
# 1.15.0, wandb 0.22.3, and the vendored TRL 1.14.0.dev0 checkout that
# prepare.sh pip-installs from the workspace.
# Tasks: algorithm/trl_dapo_accumulation_normalization,
# infrastructure/trl_conditional_reward_logging.
# Not yet built.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir setuptools==80.9.0 wheel==0.45.1 \
    && python3 -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -m pip install --no-cache-dir transformers==5.15.1 datasets==4.7.0 accelerate==1.15.0 \
       wandb==0.22.3 numpy==2.2.6 safetensors==0.8.0 tokenizers==0.22.2 peft==0.18.1 rich==14.2.0

# Transitive closure captured from the recording rig's requirements.txt (the
# wandb offline run's files/requirements.txt shipped with the control oracle).
# The rig was layered, so some packages appear twice there; the versions below
# are the ones from the layer the recorded runs used (wandb 0.22.3, datasets
# 4.7.0, accelerate 1.15.0).
# torch's own nvidia-* and triton deps are pinned by the cu124 wheel metadata
# and stay unpinned here, as in the other images.
RUN python3 -m pip install --no-cache-dir -r /dev/stdin <<'REQS'
aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiosignal==1.4.0
annotated-types==0.7.0
anyio==4.12.1
attrs==25.4.0
certifi==2026.1.4
charset-normalizer==3.4.4
click==8.5.0
dill==0.4.0
filelock==3.20.3
frozenlist==1.8.0
fsspec==2025.10.0
gitdb==4.0.12
GitPython==3.1.46
h11==0.16.0
hf-xet==1.6.0
huggingface_hub==1.29.0
httpcore==1.0.9
httpx==0.28.1
idna==3.11
Jinja2==3.1.6
markdown-it-py==4.0.0
MarkupSafe==3.0.3
mdurl==0.1.2
multidict==6.7.0
mpmath==1.3.0
multiprocess==0.70.18
networkx==3.6.1
packaging==25.0
pandas==2.3.3
platformdirs==4.5.1
protobuf==6.33.4
propcache==0.4.1
psutil==7.2.1
pyarrow==22.0.0
pydantic==2.12.5
pydantic_core==2.41.5
Pygments==2.19.2
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.3
regex==2026.1.15
requests==2.32.5
sentry-sdk==2.49.0
setproctitle==1.3.7
shellingham==1.5.4
six==1.17.0
smmap==5.0.2
sympy==1.13.1
tqdm==4.67.1
typing-inspection==0.4.2
typing_extensions==4.15.0
typer==0.21.1
tzdata==2025.3
urllib3==2.6.3
xxhash==3.6.0
yarl==1.22.0
REQS

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
