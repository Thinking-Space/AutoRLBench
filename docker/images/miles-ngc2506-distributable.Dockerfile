# PTDBench environment `miles-ngc2506` — DISTRIBUTABLE rebuild of the author's
# calibration image (business name `miles-rl-lgs`, canonical
# llm-rl/ngc-2506:nvidia.11b60366, private Aliyun ACR — not pullable by
# evaluators outside the org).
#
# Lineage (forensics 2026-11-04, inspection devspace 1457586):
#   base    = lmsysorg/sglang:v0.5.16-cu130 (created 2026-07-24)
#   native  = yueming-yuan/miles-wheels GitHub release `cu130-x86_64`
#             (2026-06-05) — install-record sha256s match the registered
#             image exactly for every shared wheel (spot-verified:
#             sglang_router, transformer_engine_torch)
#   private = sglang fork 0.5.16.dev44+g4fb9363f5 (49 files differ from
#             public d218d6c7: scheduler set + minicpm/dspark extras + H800
#             triton configs), megatron_core wheel 0.18.0+caeabd765
#             (commit not public), miles wheel 0.2.1+2a15d7b0 (source is
#             public on radixark/miles but the wheel is a compiled build),
#             infllm_v2 / sparse_kernel_extension / deep_ep 1.2.1 /
#             mscclpp 0.9.1 / hydra 1.4.0.dev1 / omegaconf 2.4.0.dev3.
#             All of these survive as installed trees in the image's
#             dist-packages and are exported verbatim into
#             docker/images/miles-ngc2506-assets/ (gitignored; sha256s in
#             assets README — for a Codeup-hosted build, upload them to an
#             HF repo and fetch at build time, cf. the search-r1 wheelhouse
#             task pattern).
#   public  = /root/Megatron-LM @ radixark/Megatron-LM 4716f754 (clean tree,
#             picked up at runtime via the workers' PYTHONPATH, shadowing the
#             dist-packages megatron wheel), /root/miles @ radixark/miles
#             c372e3b5 (reference checkout), Megatron-Bridge @ 7f0fb345,
#             torch_memory_saver @ 74d68c5e.
#
# Run-time import resolution to preserve (verified in-container):
#   import sglang    -> dist-packages (fork tree, exported)
#   import miles     -> dist-packages (wheel tree, exported; NOT /root/miles)
#   import megatron  -> /root/Megatron-LM via the train script's worker env
#                       PYTHONPATH; driver-side falls back to the dist-packages
#                       wheel tree — BOTH are shipped here.
#
# Build platform: linux/amd64 (the wheels and the base are x86_64-only).

FROM lmsysorg/sglang:v0.5.16-cu130@sha256:984699c298a95b73c469b2191403ddc85fd780506e13c39c4afff3845e27bc6c

# ---- era pins that deviate from the base image (all on PyPI, verified) ----
RUN pip install --no-deps \
      sglang-kernel==0.4.4 \
      ray==2.56.1 \
      tokenizers==0.22.2 \
      triton==3.6.0 \
      megatron-energon==7.4.0 \
      tilelang==0.1.9 \
      st_attn==0.0.7 \
      tile_kernels==1.0.0 \
      flash-attn-4==4.0.0b15 \
      apache-tvm-ffi==0.1.9 \
      sgl-deep-gemm==0.1.4 \
      nvidia-modelopt==0.45.0

# flashinfer 0.6.12 (base ships 0.6.14). flashinfer-python==0.6.12 is on PyPI;
# the +cu130 jit-cache companion comes from flashinfer's own wheel index —
# if that index is unreachable at build time, export the flashinfer* trees
# from the registered image exactly like the private assets below.
RUN pip install "flashinfer-python==0.6.12" \
      --extra-index-url https://flashinfer.ai/whl/cu130 \
 && pip install --no-deps flashinfer-jit-cache==0.6.12+cu130 \
      --extra-index-url https://flashinfer.ai/whl/cu130 \
 || pip install flashinfer-python==0.6.12

# ---- public native stack: yueming-yuan/miles-wheels release cu130-x86_64 ----
RUN set -eux; \
    mkdir -p /tmp/wheels; \
    for w in apex-0.1-cp312-cp312-linux_x86_64.whl \
             causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl \
             fake_int4_quant_cuda-0.0.0-cp312-cp312-linux_x86_64.whl \
             fast_hadamard_transform-1.1.0-cp312-cp312-linux_x86_64.whl \
             flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl \
             flash_attn_3-3.0.0-cp310-abi3-linux_x86_64.whl \
             mamba_ssm-2.3.1-cp312-cp312-linux_x86_64.whl \
             mooncake_transfer_engine_cuda13-0.3.13.dev0+g4dbe5a4c-cp312-cp312-manylinux_2_39_x86_64.whl \
             transformer_engine-2.17.0-py3-none-any.whl \
             transformer_engine_cu13-2.17.0-py3-none-manylinux_2_28_x86_64.whl \
             transformer_engine_torch-2.17.0-cp312-cp312-linux_x86_64.whl \
             sglang_router-0.3.2-cp38-abi3-manylinux_2_39_x86_64.whl; do \
      curl -fsSL -o "/tmp/wheels/$w" \
        "https://github.com/yueming-yuan/miles-wheels/releases/download/cu130-x86_64/$w"; \
    done; \
    pip install --no-deps /tmp/wheels/*.whl

# ---- public source pins ----
RUN git clone https://github.com/radixark/Megatron-LM.git /root/Megatron-LM \
 && git -C /root/Megatron-LM checkout 4716f75475c78e2fc2c6f0d3af095f1681b770b4
RUN git clone https://github.com/radixark/miles.git /root/miles \
 && git -C /root/miles checkout c372e3b50ced3c4b84f879b7530338d9db52db64
RUN pip install --no-deps \
      "git+https://github.com/radixark/Megatron-Bridge.git@7f0fb3456f8ffe47599b5fd167b454605d85f932" \
      "git+https://github.com/fzyzcjy/torch_memory_saver.git@74d68c5e4bedf2b6774f2c92ed0f81b7c8d91ed0"

# ---- private exported trees (sha256-pinned, see miles-ngc2506-assets/) ----
# Each tar is the installed dist-packages tree from the registered image,
# dist-info included, so `pip list` reproduces the original install records.
COPY miles-ngc2506-assets/sglang-0.5.16.dev44+g4fb9363f5-dist.tar.gz \
     miles-ngc2506-assets/private-megatron.tar.gz \
     miles-ngc2506-assets/private-miles.tar.gz \
     miles-ngc2506-assets/private-misc.tar.gz /tmp/
RUN set -eux; \
    test "$(sha256sum /tmp/sglang-0.5.16.dev44+g4fb9363f5-dist.tar.gz | cut -d' ' -f1)" = \
         "1d72b49fdce85493b7d86a050c9cd1fedafea4d3ed0c69fa506cb70e7721ee4c"; \
    test "$(sha256sum /tmp/private-megatron.tar.gz | cut -d' ' -f1)" = \
         "516ebd13e61b91a611dde7cbdb07de2a9b6284c72fbe27f0503728d75ca6709b"; \
    test "$(sha256sum /tmp/private-miles.tar.gz | cut -d' ' -f1)" = \
         "809f46281475c4db241dcb69e94d23b5ae087139cb07f49a173824c605df0527"; \
    test "$(sha256sum /tmp/private-misc.tar.gz | cut -d' ' -f1)" = \
         "f0f8e0d37f47b71df566766972eef986aae736278e40b73d449b97b78377f133"; \
    SD=/usr/local/lib/python3.12/dist-packages; \
    rm -rf $SD/sglang $SD/megatron $SD/miles; \
    tar -xzf /tmp/sglang-0.5.16.dev44+g4fb9363f5-dist.tar.gz -C $SD; \
    tar -xzf /tmp/private-megatron.tar.gz -C $SD; \
    tar -xzf /tmp/private-miles.tar.gz -C $SD; \
    tar -xzf /tmp/private-misc.tar.gz -C $SD; \
    rm /tmp/*.tar.gz

ENV PATH="/root/.cargo/bin:${PATH}"

# No AutoRLBench tooling is baked: the harness bind-mounts /autorlbench/prepared per
# run and injects the activate logic inline (see autorlbench/runtimes/base.py).

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TOKENIZERS_PARALLELISM=false

# The harness bind-mounts a per-run copy of the task's workspace here.
WORKDIR /workspace
