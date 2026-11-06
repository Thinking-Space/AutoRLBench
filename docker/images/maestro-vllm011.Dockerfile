# AutoRLBench environment `maestro-vllm011`.
# autorlbench-image: autorlbench/maestro-vllm011:torch280-glm016-20261105
#
# BUILD RECIPE, not a pull file (see docker/README.md "Exceptions"): the
# recording rig's two deliberately separate Python stacks -- 3.10 "maestro"
# (torch 2.8.0, vLLM 0.11.0, transformers 4.57.6) and 3.11 "maestro-glm"
# (torch 2.9.1, vLLM 0.16.0, transformers 5.2.0) -- built from source over the
# digest-pinned CUDA devel base with every package version pinned. Build and
# push it once, then collapse this file to the single digest-pinned
# `FROM autorlbench/maestro-vllm011:torch280-glm016-20261105@sha256:...`
# line the convention expects. Until that push exists, pulling the tag above
# fails -- do not run the tasks before building.
#
# Recorded runs: 2026-11-02 and 2026-11-04, 200-step multimodal GRPO.
# No host conda, weights or workspace source in the image.
# Tasks: algorithm/maestro_grpo_reward_collapse.
#
# Adapted to the bake-nothing convention (see docker/README.md "Runtime"):
# no /autorlbench mount-point mkdir (the Docker backend creates bind
# targets), no activate hook baked in (activation is injected at runtime by
# autorlbench/runtimes/base.py), and no CUDA_DEVICE_ORDER /
# TOKENIZERS_PARALLELISM ENV -- the harness seeds them at runtime. The rest
# of the recipe below is the task author's, unchanged. Build status: not yet
# built; GPU-free imports below do not establish training parity.
FROM docker.m.daocloud.io/nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=4
ENV CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=${MAX_JOBS} \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" \
    HF_HOME=/tmp/huggingface \
    HUGGINGFACE_HUB_CACHE=/tmp/huggingface/hub \
    TRANSFORMERS_CACHE=/tmp/huggingface/hub \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHON=/opt/maestro/bin/python \
    VLLM_PYTHON=/opt/maestro/bin/python \
    GLM_PYTHON=/opt/maestro-glm/bin/python \
    WANDB_MODE=offline \
    PATH=/opt/maestro/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git xz-utils pkg-config ninja-build cmake \
    libssl-dev zlib1g-dev libffi-dev libbz2-dev libreadline-dev libsqlite3-dev \
    liblzma-dev libgdbm-dev libgdbm-compat-dev libncurses-dev uuid-dev \
    libnuma-dev libibverbs-dev librdmacm-dev libaio-dev libsndfile1 \
    libgl1 libglib2.0-0 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*

# SHA-256 digests from the python.org release Sigstore bundles.
RUN set -eux; \
    for spec in \
      '3.10.20:maestro:de6517421601e39a9a3bc3e1bc4c7b2f239297423ee05e282598c83ec0647505' \
      '3.11.14:maestro-glm:8d3ed8ec5c88c1c95f5e558612a725450d2452813ddad5e58fdb1a53b1209b78'; do \
      version="${spec%%:*}"; rest="${spec#*:}"; name="${rest%%:*}"; digest="${rest#*:}"; \
      curl -fSL --retry 3 "https://www.python.org/ftp/python/$version/Python-$version.tar.xz" -o /tmp/python.tar.xz; \
      printf '%s  %s\n' "$digest" /tmp/python.tar.xz | sha256sum -c -; \
      mkdir /tmp/python-source; \
      tar -xJf /tmp/python.tar.xz -C /tmp/python-source --strip-components=1; \
      cd /tmp/python-source; \
      ./configure --prefix="/opt/$name" --with-ensurepip=install; \
      make -j"$MAX_JOBS"; make install; \
      ln -s python3 "/opt/$name/bin/python"; \
      "/opt/$name/bin/python" -c "import ssl, sqlite3, bz2, lzma, ctypes, sys; assert sys.version.split()[0] == '$version'"; \
      cd /; rm -rf /tmp/python-source /tmp/python.tar.xz; \
    done

# Embed pins: the standard activate hook is the only local COPY input.
RUN /opt/maestro/bin/python - <<'PY'
from pathlib import Path
import subprocess
import tempfile
requirements = {
    'maestro': 'absl-py==2.4.0\naccelerate==1.13.0\nacecoder==0.0.1\naiofiles==25.1.0\naiohappyeyeballs==2.6.1\naiohttp-cors==0.8.1\naiohttp==3.13.3\naiosignal==1.4.0\nannotated-doc==0.0.4\nannotated-types==0.7.0\nanthropic==0.86.0\nantlr4-python3-runtime==4.9.3\nanyio==4.12.1\napache-tvm-ffi==0.1.9\nappdirs==1.4.4\nastor==0.8.1\nasttokens==3.0.1\nasync-timeout==5.0.1\nattrs==26.1.0\naudioread==3.1.0\nav==17.0.0\nbeautifulsoup4==4.14.3\nblake3==1.0.8\nbs4==0.0.2\ncachetools==7.0.5\ncbor2==5.9.0\ncertifi==2026.2.25\ncffi==2.0.0\ncharset-normalizer==3.4.6\nclick==8.2.1\ncloudpickle==3.1.2\ncodetiming==1.4.0\ncolorama==0.4.6\ncolorful==0.5.8\ncompressed-tensors==0.11.0\ncontourpy==1.3.2\ncryptography==46.0.5\ncuda-bindings==12.9.4\ncuda-pathfinder==1.4.4\ncuda-python==12.9.4\ncuda-toolkit==13.0.2\ncupy-cuda12x==14.0.1\ncycler==0.12.1\ndatasets==4.8.4\ndecorator==5.2.1\ndepyf==0.19.0\ndill==0.4.1\ndiskcache==5.6.3\ndistlib==0.4.0\ndistro==1.9.0\ndnspython==2.8.0\ndocstring_parser==0.17.0\neinops==0.8.2\nemail-validator==2.3.0\nevalplus==0.3.1\nexceptiongroup==1.3.1\nexecuting==2.2.1\nfastapi-cli==0.0.24\nfastapi-cloud-cli==0.15.0\nfastapi==0.135.2\nfastar==0.9.0\nfilelock==3.25.2\nfire==0.7.1\nflash_attn==2.8.3\nflashinfer-python==0.6.3\nfonttools==4.63.0\nfrozendict==2.4.7\nfrozenlist==1.8.0\nfsspec==2026.2.0\nfunc_timeout==4.3.5\ngguf==0.18.0\ngitdb==4.0.12\nGitPython==3.1.46\ngoogle-ai-generativelanguage==0.6.15\ngoogle-api-core==2.25.0\ngoogle-api-python-client==2.171.0\ngoogle-auth-httplib2==0.2.0\ngoogle-auth==2.49.1\ngoogle-genai==1.19.0\ngoogle-generativeai==0.8.6\ngoogle==3.0.0\ngoogleapis-common-protos==1.73.0\ngrpcio-reflection==1.78.0\ngrpcio-status==1.71.2\ngrpcio==1.78.0\nh11==0.16.0\nhf-xet==1.4.2\nhf_transfer==0.1.9\nhishel==1.1.9\nhttpcore==1.0.9\nhttplib2==0.31.2\nhttptools==0.7.1\nhttpx-sse==0.4.3\nhttpx==0.28.1\nhuggingface_hub==0.36.2\nhydra-core==1.3.2\nidna==3.11\nijson==3.5.0\nimportlib_metadata==8.7.1\ninteregular==0.3.3\nipython==8.38.0\njedi==0.19.2\nJinja2==3.1.6\njiter==0.13.0\njmespath==1.1.0\njoblib==1.5.3\njsonschema-specifications==2025.9.1\njsonschema==4.26.0\nkiwisolver==1.5.0\nlangid==1.1.6\nlark==1.2.2\nlatex2sympy2_extended==1.11.0\nlazy-loader==0.5\nlibrosa==0.11.0\nllguidance==0.7.30\nllvmlite==0.44.0\nlm-format-enforcer==0.11.3\nloguru==0.7.3\nlxml==6.0.2\nmarkdown-it-py==4.0.0\nMarkdown==3.10.2\nMarkupSafe==3.0.3\nmath-verify==0.9.0\nmatplotlib-inline==0.2.1\nmatplotlib==3.10.9\nmcp==1.26.0\nmdurl==0.1.2\nmistral_common==1.10.0\nmodel-hosting-container-standards==0.1.14\nmpmath==1.3.0\nmsgpack==1.1.2\nmsgspec==0.20.0\nmultidict==6.7.1\nmultipledispatch==1.0.0\nmultiprocess==0.70.19\nnetworkx==3.4.2\nninja==1.13.0\nnltk==3.9.4\nnumba==0.61.2\nnumpy==1.26.0\nnvidia-cublas-cu12==12.8.4.1\nnvidia-cublas==13.1.0.3\nnvidia-cuda-cupti-cu12==12.8.90\nnvidia-cuda-cupti==13.0.85\nnvidia-cuda-nvrtc-cu12==12.8.93\nnvidia-cuda-nvrtc==13.0.88\nnvidia-cuda-runtime-cu12==12.8.90\nnvidia-cuda-runtime==13.0.96\nnvidia-cudnn-cu12==9.10.2.21\nnvidia-cudnn-cu13==9.19.0.56\nnvidia-cudnn-frontend==1.18.0\nnvidia-cufft-cu12==11.3.3.83\nnvidia-cufft==12.0.0.61\nnvidia-cufile-cu12==1.13.1.3\nnvidia-cufile==1.15.1.6\nnvidia-curand-cu12==10.3.9.90\nnvidia-curand==10.4.0.35\nnvidia-cusolver-cu12==11.7.3.90\nnvidia-cusolver==12.0.4.66\nnvidia-cusparse-cu12==12.5.8.93\nnvidia-cusparse==12.6.3.3\nnvidia-cusparselt-cu12==0.7.1\nnvidia-cusparselt-cu13==0.8.0\nnvidia-cutlass-dsl-libs-base==4.4.2\nnvidia-cutlass-dsl==4.4.2\nnvidia-ml-py==13.595.45\nnvidia-nccl-cu12==2.27.3\nnvidia-nccl-cu13==2.28.9\nnvidia-nvjitlink-cu12==12.8.93\nnvidia-nvjitlink==13.0.88\nnvidia-nvshmem-cu12==3.3.20\nnvidia-nvshmem-cu13==3.4.5\nnvidia-nvtx-cu12==12.8.90\nnvidia-nvtx==13.0.85\nomegaconf==2.3.0\nopenai-harmony==0.0.8\nopenai==2.24.0\nopencensus-context==0.1.3\nopencensus==0.11.4\nopencv-python-headless==4.13.0.92\nopentelemetry-api==1.40.0\nopentelemetry-exporter-otlp-proto-common==1.40.0\nopentelemetry-exporter-otlp-proto-grpc==1.40.0\nopentelemetry-exporter-otlp-proto-http==1.40.0\nopentelemetry-exporter-otlp==1.40.0\nopentelemetry-exporter-prometheus==0.61b0\nopentelemetry-proto==1.40.0\nopentelemetry-sdk==1.40.0\nopentelemetry-semantic-conventions-ai==0.5.0\nopentelemetry-semantic-conventions==0.61b0\norjson==3.11.7\noutlines_core==0.2.11\npackaging==25.0\npandas==2.3.3\nparso==0.8.6\npartial-json-parser==0.2.1.1.post7\npdfminer.six==20251230\npdfplumber==0.11.9\npeft==0.18.1\npexpect==4.9.0\npillow==12.1.1\npip==26.0.1\nplatformdirs==4.9.4\npooch==1.9.0\nprometheus-fastapi-instrumentator==7.1.0\nprometheus_client==0.24.1\nprompt_toolkit==3.0.52\npropcache==0.4.1\nproto-plus==1.27.1\nprotobuf==5.29.6\npsutil==7.2.2\npsutils==3.3.11\nptyprocess==0.7.0\npure_eval==0.2.3\npuremagic==1.30\npy-cpuinfo==9.0.0\npy-spy==0.4.1\npyarrow==23.0.1\npyasn1==0.6.3\npyasn1_modules==0.4.2\npybase64==1.4.3\npybind11==3.0.2\npycountry==26.2.16\npycparser==3.0\npydantic-extra-types==2.11.1\npydantic-settings==2.13.1\npydantic==2.12.5\npydantic_core==2.41.5\npyext==0.7\nPygments==2.19.2\nPyJWT==2.12.1\npylatexenc==2.10\npyparsing==3.3.2\npypdf==6.9.2\npypdfium2==5.6.0\npython-dateutil==2.9.0.post0\npython-discovery==1.2.0\npython-dotenv==1.2.2\npython-json-logger==4.0.0\npython-multipart==0.0.22\npytz==2026.1.post1\npyvers==0.1.0\nPyYAML==6.0.3\npyzmq==27.1.0\nquack-kernels==0.3.5\nqwen-omni-utils==0.0.9\nqwen-vl-utils==0.0.14\nray==2.54.0\nreferencing==0.37.0\nregex==2026.2.28\nrequests==2.32.5\nrich-toolkit==0.19.7\nrich==14.3.3\nrignore==0.7.6\nrpds-py==0.30.0\nsafetensors==0.7.0\nscikit-learn==1.7.2\nscipy==1.15.3\nsentencepiece==0.2.1\nsentry-sdk==2.55.0\nsetproctitle==1.3.7\nsetuptools==80.10.2\nshellingham==1.5.4\nsix==1.17.0\nsmart_open==7.5.1\nsmmap==5.0.3\nsniffio==1.3.1\nsoundfile==0.13.1\nsoupsieve==2.8.3\nsoxr==1.0.0\nsse-starlette==3.3.3\nstack-data==0.6.3\nstarlette==0.52.1\nstop-sequencer==1.2.3\nsupervisor==4.3.0\nsympy==1.14.0\ntabulate==0.10.0\ntempdir==0.7.1\ntensorboard-data-server==0.7.2\ntensorboard==2.20.0\ntensordict==0.10.0\ntermcolor==3.3.0\nthreadpoolctl==3.6.0\ntiktoken==0.12.0\ntimeout-decorator==0.5.0\ntokenizers==0.22.2\ntomli==2.4.0\ntorch==2.8.0\ntorch_c_dlpack_ext==0.1.5\ntorchaudio==2.8.0\ntorchdata==0.11.0\ntorchvision==0.23.0\ntqdm==4.67.3\ntraitlets==5.14.3\ntransformers==4.57.6\ntree-sitter-python==0.25.0\ntree-sitter==0.25.2\ntriton==3.4.0\ntyper==0.24.1\ntyping-inspection==0.4.2\ntyping_extensions==4.15.0\ntzdata==2025.3\nuritemplate==4.2.0\nurllib3==2.6.3\nuvicorn==0.42.0\nuvloop==0.22.1\nvirtualenv==21.2.0\nvllm==0.11.0\nwandb==0.25.1\nwatchfiles==1.1.1\nwcwidth==0.6.0\nwebsockets==15.0.1\nWerkzeug==3.1.7\nwget==3.2\nwheel==0.46.3\nwrapt==2.1.2\nxformers==0.0.32.post1\nxgrammar==0.1.25\nxxhash==3.6.0\nyarl==1.23.0\nzipp==3.23.0\n',
    'maestro-glm': 'aiohappyeyeballs==2.6.1\naiohttp==3.13.3\naiosignal==1.4.0\nannotated-doc==0.0.4\nannotated-types==0.7.0\nanthropic==0.84.0\nanyio==4.12.1\napache-tvm-ffi==0.1.9\nastor==0.8.1\nattrs==25.4.0\nblake3==1.0.8\ncachetools==7.0.1\ncbor2==5.8.0\ncertifi==2026.2.25\ncffi==2.0.0\ncharset-normalizer==3.4.4\nclick==8.3.1\ncloudpickle==3.1.2\ncompressed-tensors==0.13.0\ncryptography==46.0.5\ncuda-bindings==13.1.1\ncuda-pathfinder==1.4.0\ncuda-python==13.1.1\ncupy-cuda12x==14.0.1\ndepyf==0.20.0\ndill==0.4.1\ndiskcache==5.6.3\ndistro==1.9.0\ndnspython==2.8.0\ndocstring_parser==0.17.0\neinops==0.8.2\nemail-validator==2.3.0\nfastapi-cli==0.0.24\nfastapi-cloud-cli==0.14.0\nfastapi==0.135.0\nfastar==0.8.0\nfilelock==3.24.3\nflashinfer-python==0.6.3\nfrozenlist==1.8.0\nfsspec==2026.2.0\ngguf==0.18.0\ngrpcio-reflection==1.78.0\ngrpcio==1.78.0\nh11==0.16.0\nhf-xet==1.3.2\nhttpcore==1.0.9\nhttptools==0.7.1\nhttpx-sse==0.4.3\nhttpx==0.28.1\nhuggingface_hub==1.5.0\nidna==3.11\nijson==3.5.0\ninteregular==0.3.3\nJinja2==3.1.6\njiter==0.13.0\njmespath==1.1.0\njsonschema-specifications==2025.9.1\njsonschema==4.26.0\nlark==1.2.2\nllguidance==1.3.0\nllvmlite==0.44.0\nlm-format-enforcer==0.11.3\nloguru==0.7.3\nmarkdown-it-py==4.0.0\nMarkupSafe==3.0.3\nmcp==1.26.0\nmdurl==0.1.2\nmistral_common==1.9.1\nmodel-hosting-container-standards==0.1.13\nmpmath==1.3.0\nmsgpack==1.1.2\nmsgspec==0.20.0\nmultidict==6.7.1\nnetworkx==3.6.1\nninja==1.13.0\nnumba==0.61.2\nnumpy==2.2.6\nnvidia-cublas-cu12==12.8.4.1\nnvidia-cuda-cupti-cu12==12.8.90\nnvidia-cuda-nvrtc-cu12==12.8.93\nnvidia-cuda-runtime-cu12==12.8.90\nnvidia-cudnn-cu12==9.10.2.21\nnvidia-cudnn-frontend==1.18.0\nnvidia-cufft-cu12==11.3.3.83\nnvidia-cufile-cu12==1.13.1.3\nnvidia-curand-cu12==10.3.9.90\nnvidia-cusolver-cu12==11.7.3.90\nnvidia-cusparse-cu12==12.5.8.93\nnvidia-cusparselt-cu12==0.7.1\nnvidia-cutlass-dsl-libs-base==4.4.1\nnvidia-cutlass-dsl==4.4.1\nnvidia-ml-py==13.590.48\nnvidia-nccl-cu12==2.27.5\nnvidia-nvjitlink-cu12==12.8.93\nnvidia-nvshmem-cu12==3.3.20\nnvidia-nvtx-cu12==12.8.90\nopenai-harmony==0.0.8\nopenai==2.24.0\nopencv-python-headless==4.13.0.92\noutlines_core==0.2.11\npackaging==25.0\npartial-json-parser==0.2.1.1.post7\npillow==12.1.1\npip==26.0.1\nprometheus-fastapi-instrumentator==7.1.0\nprometheus_client==0.24.1\npropcache==0.4.1\nprotobuf==6.33.5\npsutil==7.2.2\npy-cpuinfo==9.0.0\npybase64==1.4.3\npycountry==26.2.16\npycparser==3.0\npydantic-extra-types==2.11.0\npydantic-settings==2.13.1\npydantic==2.12.5\npydantic_core==2.41.5\nPygments==2.19.2\nPyJWT==2.11.0\npython-dotenv==1.2.1\npython-json-logger==4.0.0\npython-multipart==0.0.22\nPyYAML==6.0.3\npyzmq==27.1.0\nray==2.54.0\nreferencing==0.37.0\nregex==2026.2.28\nrequests==2.32.5\nrich-toolkit==0.19.7\nrich==14.3.3\nrignore==0.7.6\nrpds-py==0.30.0\nsafetensors==0.7.0\nsentencepiece==0.2.1\nsentry-sdk==2.53.0\nsetproctitle==1.3.7\nsetuptools==80.10.2\nshellingham==1.5.4\nsniffio==1.3.1\nsse-starlette==3.3.2\nstarlette==0.52.1\nsupervisor==4.3.0\nsympy==1.14.0\ntabulate==0.9.0\ntiktoken==0.12.0\ntokenizers==0.22.2\ntorch==2.9.1\ntorchaudio==2.9.1\ntorchvision==0.24.1\ntqdm==4.67.3\ntransformers==5.2.0\ntriton==3.5.1\ntyper-slim==0.24.0\ntyper==0.24.1\ntyping-inspection==0.4.2\ntyping_extensions==4.15.0\nurllib3==2.6.3\nuvicorn==0.41.0\nuvloop==0.22.1\nvllm==0.16.0\nwatchfiles==1.1.1\nwebsockets==16.0\nwheel==0.46.3\nxgrammar==0.1.29\nyarl==1.22.0\n',
}
for name, content in requirements.items():
    python = '/opt/' + name + '/bin/python'
    lines = content.splitlines()
    key = lambda line: line.split('==')[0].lower().replace('_', '-')
    bootstrap = [line for line in lines if key(line) in {'pip', 'setuptools', 'wheel'}]
    flash = [line for line in lines if key(line) == 'flash-attn']
    ordinary = [line for line in lines if line not in flash]
    subprocess.run([python, '-m', 'pip', 'install', '--no-deps', *bootstrap], check=True)
    with tempfile.TemporaryDirectory(prefix='maestro-image-pins-') as temporary:
        path = Path(temporary) / 'requirements.txt'
        path.write_text('\n'.join(ordinary) + '\n')
        subprocess.run([python, '-m', 'pip', 'install', '--no-deps', '-r', str(path)], check=True)
    if flash:
        subprocess.run([python, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', *flash], check=True)
PY

# No workspace imports or training happen at build time.
RUN /opt/maestro/bin/python -c "import torch, ray, transformers, flash_attn, pyarrow, wandb; import importlib.metadata as m; assert torch.__version__.split('+')[0]=='2.8.0'; assert m.version('vllm')=='0.11.0'; assert m.version('transformers')=='4.57.6'; assert m.version('ray')=='2.54.0'" && \
    /opt/maestro-glm/bin/python -c "import torch, transformers; import importlib.metadata as m; assert torch.__version__.split('+')[0]=='2.9.1'; assert m.version('vllm')=='0.16.0'; assert m.version('transformers')=='5.2.0'"
