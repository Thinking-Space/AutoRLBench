# AutoRLBench environment `verl-moe-token`.
# autorlbench-image: autorlbench/verl-moe-token:vllm0.17-torch2.10-cu129-mcore0.16
#
# The shared `verl` image (same upstream digest) plus one Megatron-Core router
# patch. Stack: torch 2.10.0+cu129, vLLM 0.17.0, megatron-core 0.16.0, mbridge
# 0.15.1, transformer-engine 2.12.0, transformers 4.57.6, ray 2.54.0 -- see
# docker/verl-moe-token/image.json for the image the task was recorded on.
# Tasks: infrastructure/verl_moe_token_normalization.
#
# The patch (docker/verl-moe-token/mcore5047-minimal.patch) backports MCore's
# tp_cp_group scaling of the MoE aux/z-loss in router.py. The recorded runs were
# produced with it applied; it is shared by the defective and repaired arms and
# is not the defect under diagnosis.
#
# UNVERIFIED: this Dockerfile has not been built yet. At first build, compare the
# patched router.py sha256 with `router_sha256` in image.json.
FROM verlai/verl:vllm017.latest@sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6

COPY verl-moe-token/mcore5047-minimal.patch /tmp/mcore5047-minimal.patch
RUN set -eux; \
    mcore="$(python3 -c 'import megatron.core, os; print(os.path.dirname(os.path.dirname(megatron.core.__file__)))')"; \
    patch -d "$mcore" -p1 --forward < /tmp/mcore5047-minimal.patch; \
    rm /tmp/mcore5047-minimal.patch; \
    sha256sum "$mcore/megatron/core/transformer/moe/router.py"; \
    python3 -c "import megatron.core; print('megatron-core', megatron.core.__version__)"

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
