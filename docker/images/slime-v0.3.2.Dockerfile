# AutoRLBench environment `slime-v0.3.2`.
# autorlbench-image: autorlbench/slime:v0.3.2-sglang0.5.15-cu129
#
# slime v0.3.2 stack: sglang 0.5.15.post1, CUDA 12.9, Ubuntu 24.04.
# Tasks: slime_retool, slime_retool_length_hack, slime_search_r1,
# slime_search_r1_nomask.
#
# The upstream image is used as-is; the only thing added is the AutoRLBench
# tooling block below. The FROM is pinned by digest because the tag moves
# upstream and the tasks' recorded outcomes must not move with it.
FROM slimerl/slime:v0.3.2-cu129@sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c

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
