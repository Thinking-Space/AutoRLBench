# AutoRLBench shared historical IDEA/AWM runtime.
# autorlbench-image: autorlbench/idea-historical:py3120-awm31213-torch280-cu128-20261023
#
# Captured run: awm_opsd_qwen3_8b_step693_wandb_v5, 2026-10-11.
# Isolated source-built CPython 3.12.0 (IDEA) and 3.12.13 (AWM).
# Exact captured requirements: torch 2.8.0, vLLM 0.11.0, flash-attn
# 2.7.4.post1, Ray 2.50.0 and the captured Transformers Git commit.
# No host conda, model weights, workspace source, agent command or training run
# is baked in. The default benchmark environment remains idea-issue.
# Build status: not yet built; a definition is not a validated GPU rerun.
FROM docker.m.daocloud.io/nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=4
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.0 \
    MAX_JOBS=${MAX_JOBS} \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WANDB_MODE=offline \
    TRAIN_PYTHON=/opt/idea/bin/python \
    AWM_PYTHON=/opt/awm/bin/python \
    PYTHONPATH=/workspace/idea_source \
    PATH=/opt/idea/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git xz-utils pkg-config ninja-build cmake \
        libssl-dev zlib1g-dev libffi-dev libbz2-dev libreadline-dev libsqlite3-dev \
        liblzma-dev libgdbm-dev libgdbm-compat-dev libncurses-dev uuid-dev \
        libnuma-dev libibverbs-dev librdmacm-dev libaio-dev libsndfile1 \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# SHA-256 values are the message digests in python.org's release Sigstore
# bundles for Python-3.12.0.tar.xz and Python-3.12.13.tar.xz.
# Independent prefixes (no shared libpython) prevent interpreter/library mixing.
RUN set -eux; \
    for spec in \
      '3.12.0:idea:795c34f44df45a0e9b9710c8c71c15c671871524cd412ca14def212e8ccb155d' \
      '3.12.13:awm:c08bc65a81971c1dd5783182826503369466c7e67374d1646519adf05207b684'; do \
        version="${spec%%:*}"; rest="${spec#*:}"; name="${rest%%:*}"; digest="${rest#*:}"; \
        curl -fSL --retry 3 "https://www.python.org/ftp/python/$version/Python-$version.tar.xz" -o /tmp/python.tar.xz; \
        printf '%s  %s\n' "$digest" /tmp/python.tar.xz | sha256sum -c -; \
        mkdir /tmp/python-source; \
        tar -xJf /tmp/python.tar.xz -C /tmp/python-source --strip-components=1; \
        cd /tmp/python-source; \
        ./configure --prefix="/opt/$name" --with-ensurepip=install; \
        make -j"$MAX_JOBS"; \
        make install; \
        ln -s python3 "/opt/$name/bin/python"; \
        "/opt/$name/bin/python" -c "import ssl, sqlite3, bz2, lzma, ctypes, sys; assert sys.version.split()[0] == '$version'"; \
        cd /; \
        rm -rf /tmp/python-source /tmp/python.tar.xz; \
    done

# Captured pins are embedded so this image needs no task-local files or
# separate dependency manifests in its Docker build context.
RUN /opt/idea/bin/python - <<'PY'
from pathlib import Path

requirements = {
    'idea': r"""accelerate==1.14.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiohttp-cors==0.8.1
aiosignal==1.4.0
annotated-doc==0.0.5
annotated-types==0.8.0
antlr4-python3-runtime==4.9.3
anyio==4.14.2
astor==0.8.1
attrs==26.1.0
av==18.1.0
blake3==1.0.9
cachetools==7.1.7
cbor2==6.1.4
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.1
click==8.4.2
cloudpickle==3.1.2
codetiming==1.4.0
colorful==0.5.8
compressed-tensors==0.11.0
contourpy==1.3.3
cryptography==50.0.0
cuda-pathfinder==1.6.1
cupy-cuda12x==14.2.0
cycler==0.12.1
datasets==5.0.1
decord==0.6.0
depyf==0.19.0
detect-installer==0.1.0
dill==0.4.1
diskcache==5.6.3
distlib==0.4.3
dnspython==2.8.0
einops==0.8.2
email-validator==2.3.0
fastapi==0.141.1
fastapi-cli==0.0.32
fastapi-cloud-cli==0.23.0
fastar==0.12.0
filelock==3.32.3
fire==0.7.1
flash_attn==2.7.4.post1
fonttools==4.65.0
frozendict==2.4.7
frozenlist==1.8.0
fsspec==2026.6.0
gguf==0.19.0
google-api-core==2.34.0
google-auth==2.56.3
googleapis-common-protos==1.75.1
grpcio==1.83.0
h11==0.16.0
hf-xet==1.6.0
hishel==1.3.1
httpcore==1.0.9
httpcore2==2.12.0
httptools==0.8.0
httpx==0.28.1
httpx2==2.12.0
huggingface_hub==1.28.0
hydra-core==1.3.5
idna==3.19
importlib_metadata==9.0.0
interegular==0.3.3
Jinja2==3.1.6
jiter==0.16.0
json_repair==0.63.4
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.1
lark==1.2.2
llguidance==0.7.30
llvmlite==0.44.0
lm-format-enforcer==0.11.3
loguru==0.7.3
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.11.1
mdurl==0.1.2
mistral_common==1.11.7
mpmath==1.3.0
msgpack==1.2.1
msgspec==0.21.1
multidict==6.7.1
multiprocess==0.70.19
networkx==3.6.1
ninja==1.13.0
numba==0.61.2
numpy==2.2.6
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.3
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.8.90
omegaconf==2.3.1
openai==3.3.1
openai-harmony==0.0.8
opencensus==0.11.4
opencensus-context==0.1.3
opencv-python-headless==5.0.0.93
opentelemetry-api==1.44.0
opentelemetry-exporter-prometheus==0.65b0
opentelemetry-proto==1.44.0
opentelemetry-sdk==1.44.0
opentelemetry-semantic-conventions==0.65b0
orjson==3.12.0
outlines_core==0.2.11
packaging==25.0
pandas==3.0.5
partial-json-parser==0.2.1.1.post7
peft==0.20.0
pillow==12.3.0
pip==26.1.2
platformdirs==4.11.3
prometheus-fastapi-instrumentator==8.1.0
prometheus_client==0.26.0
propcache==0.5.2
proto-plus==1.28.3
protobuf==7.36.0
psutil==7.2.2
py-cpuinfo==9.0.0
py-spy==0.4.2
pyarrow==25.0.1
pyasn1==0.6.4
pyasn1_modules==0.4.2
pybase64==1.5.0
pybind11==3.1.0
pycountry==26.2.16
pycparser==3.0
pydantic==2.13.4
pydantic-extra-types==2.11.1
pydantic-settings==2.15.0
pydantic_core==2.46.4
Pygments==2.21.0
PyJWT==2.8.0
pylatexenc==2.11
pyparsing==3.3.2
python-dateutil==2.9.0.post0
python-discovery==1.5.2
python-dotenv==1.2.3
python-json-logger==4.2.0
python-multipart==0.0.32
pyvers==0.1.0
PyYAML==6.0.3
pyzmq==27.2.0
qwen-vl-utils==0.0.14
ray==2.50.0
referencing==0.37.0
regex==2026.7.19
requests==2.34.2
rich==15.0.0
rich-toolkit==0.20.3
rignore==0.8.1
rpds-py==2026.6.3
safetensors==0.8.0
scipy==1.18.1
sentencepiece==0.2.2
sentry-sdk==2.68.0
setproctitle==1.3.7
setuptools==79.0.1
shellingham==1.5.4
six==1.17.0
smart_open==8.0.1
sniffio==1.3.1
soundfile==0.14.0
soxr==1.1.0
starlette==1.6.0
sympy==1.14.0
tensordict==0.10.0
termcolor==3.3.0
tiktoken==0.14.0
tokenizers==0.23.1
torch==2.8.0
torchaudio==2.8.0
torchdata==0.11.0
torchvision==0.23.0
tqdm==4.70.0
transformers @ git+https://github.com/huggingface/transformers.git@d56c55bf564ddb176759eb6ec199442682564916
triton==3.4.0
truststore==0.10.4
typer==0.27.1
typing-inspection==0.4.4
typing_extensions==4.16.0
urllib3==2.7.0
uvicorn==0.52.4
uvloop==0.22.1
virtualenv==21.7.4
vllm==0.11.0
wandb==0.28.2
watchfiles==1.2.0
websockets==17.0.1
wheel==0.47.0
wrapt==2.3.0
xformers==0.0.32.post1
xgrammar==0.1.25
xxhash==4.0.1
yarl==1.24.5
zai-sdk==0.0.3.3
zipp==4.1.0
""",
    'awm': r"""addict==2.4.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiosignal==1.4.0
amqp==5.3.1
annotated-types==0.7.0
anthropic==0.104.0
anyio==4.12.1
appdirs==1.4.4
argcomplete==3.6.3
argon2-cffi==25.1.0
argon2-cffi-bindings==25.1.0
attrs==25.4.0
backoff==2.2.1
bcrypt==5.0.0
beautifulsoup4==4.14.3
billiard==4.2.4
black==26.5.1
boto3==1.43.13
botocore==1.43.13
cachebox==5.2.3
celery==5.6.3
certifi==2026.1.4
cffi==2.0.0
charset-normalizer==3.4.4
claude-code-sdk==0.0.25
click==8.3.1
click-didyoumean==0.3.1
click-plugins==1.1.1.2
click-repl==0.3.0
cloudpickle==3.1.2
cohere==5.18.0
contourpy==1.3.3
cryptography==46.0.4
curl_cffi==0.15.0
cycler==0.12.1
dashscope==1.25.18
datamodel-code-generator==0.25.7
deepdiff==9.1.0
distro==1.9.0
dnspython==2.8.0
docstring_parser==0.18.0
dotenv==0.9.9
email-validator==2.3.0
eval_type_backport==0.3.1
Farama-Notifications==0.0.6
fastapi==0.115.12
fastapi-mcp==0.4.0
fastavro==1.12.2
fastuuid==0.14.0
filelock==3.20.0
fonttools==4.63.0
frozendict==2.4.7
frozenlist==1.8.0
fs==2.4.16
fsspec==2026.4.0
genson==1.3.0
google-api-core==2.30.3
google-api-python-client==2.196.0
google-auth==2.53.0
google-auth-httplib2==0.4.0
google-auth-oauthlib==1.4.0
google-genai==2.5.0
google_search_results==2.4.2
googleapis-common-protos==1.72.0
greenlet==3.3.1
griffelib==2.0.2
grpcio==1.80.0
gymnasium==1.3.0
h11==0.16.0
hf-xet==1.5.0
html2text==2025.4.15
html5lib==1.1
httpcore==1.0.9
httplib2==0.31.2
httptools==0.7.1
httpx==0.28.1
httpx-sse==0.4.0
huggingface_hub==1.16.1
idna==3.11
importlib_metadata==8.7.1
inflect==5.6.2
iniconfig==2.3.0
iso8601==2.1.0
isort==5.13.2
Jinja2==3.1.6
jiter==0.13.0
jmespath==1.1.0
joblib==1.5.3
json5==0.14.0
json_repair==0.56.0
jsonlines==4.0.0
jsonref==1.1.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kafka-python==2.3.1
kiwisolver==1.5.0
kombu==5.6.2
langfuse==4.6.1
litellm==1.85.1
loguru==0.7.3
lxml==6.1.1
markdown-it-py==4.0.0
markdownify==1.2.2
MarkupSafe==3.0.3
matplotlib==3.10.9
mcp==1.26.0
mcp-agent==0.2.6
mcp-server-calculator==0.2.0
mcp-server-fetch==2025.1.17
mdurl==0.1.2
mistralai==1.7.0
mpmath==1.3.0
multidict==6.7.1
multitasking==0.0.13
mypy_extensions==1.1.0
narwhals==2.21.2
networkx==3.3
numpy==2.4.2
oauthlib==3.3.1
openai==2.38.0
openai-agents==0.17.3
opentelemetry-api==1.39.1
opentelemetry-distro==0.60b1
opentelemetry-exporter-otlp-proto-common==1.39.1
opentelemetry-exporter-otlp-proto-http==1.39.1
opentelemetry-instrumentation==0.60b1
opentelemetry-instrumentation-anthropic==0.52.1
opentelemetry-instrumentation-openai==0.52.1
opentelemetry-proto==1.39.1
opentelemetry-sdk==1.39.1
opentelemetry-semantic-conventions==0.60b1
opentelemetry-semantic-conventions-ai==0.4.13
orderly-set==5.5.0
orjson==3.11.7
overrides==7.7.0
packaging==25.0
pandas==3.0.3
pathspec==1.0.4
peewee==4.0.6
pika==1.4.0
pillow==12.2.0
platformdirs==4.9.6
playwright==1.60.0
plotly==6.7.0
pluggy==1.6.0
portalocker==3.2.0
prompt_toolkit==3.0.52
propcache==0.4.1
Protego==0.6.0
proto-plus==1.28.0
protobuf==6.33.5
psutil==7.2.2
psycopg==3.3.4
psycopg-binary==3.3.4
pyasn1==0.6.3
pyasn1_modules==0.4.2
pycparser==3.0
pycryptodomex==3.23.0
pydantic==2.12.5
pydantic-argparse==0.10.0
pydantic-settings==2.12.0
pydantic-yaml==1.6.0
pydantic_core==2.41.5
pyee==13.0.1
Pygments==2.19.2
PyJWT==2.11.0
pyparsing==3.3.2
pyseto==1.9.3
pytest==9.0.3
python-dateutil==2.9.0.post0
python-dotenv==1.2.1
python-multipart==0.0.22
pytokens==0.4.1
pytz==2026.2
PyYAML==6.0.3
qwen-agent==0.0.34
rank-bm25==0.2.2
readabilipy==0.3.0
redis==7.4.0
referencing==0.37.0
regex==2026.1.15
requests==2.32.5
requests-oauthlib==2.0.0
rich==14.3.2
rpds-py==0.30.0
ruamel.yaml==0.18.17
ruamel.yaml.clib==0.2.15
ruff==0.15.14
s3transfer==0.17.0
schema==0.7.8
scikit-learn==1.8.0
scipy==1.17.0
seaborn==0.13.2
setuptools==82.0.1
shellingham==1.5.4
simpleArgParser==0.2.2
six==1.17.0
sniffio==1.3.1
soundfile==0.13.1
soupsieve==2.8.3
SQLAlchemy==2.0.41
sse-starlette==3.0.3
starlette==0.46.2
tabulate==0.10.0
tenacity==9.1.4
threadpoolctl==3.6.0
tiktoken==0.12.0
tokenizers==0.23.1
toml==0.10.2
tomli==2.4.0
tqdm==4.67.3
tree-sitter==0.21.3
tree-sitter-java==0.21.0
tree-sitter-javascript==0.21.4
typer==0.21.1
types-requests==2.33.0.20260518
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2026.2
tzlocal==5.3.1
uritemplate==4.2.0
urllib3==2.6.3
uvicorn==0.40.0
uvloop==0.22.1
vine==5.1.0
watchdog==6.0.0
watchfiles==1.2.0
wcwidth==0.5.3
webencodings==0.5.1
websocket-client==1.9.0
websockets==16.0
Wikipedia-API==0.14.0
wrapt==1.17.3
writer-sdk==2.4.0
xai-sdk==1.12.2
yarl==1.22.0
yfinance==1.3.0
zipp==3.23.0
""",
}
root = Path('/opt/idea-runtime')
root.mkdir(parents=True, exist_ok=True)
for name, content in requirements.items():
    (root / (name + '-requirements.txt')).write_text(content)
PY
# Install from captured package pins without resolving a replacement torch or
# combining the intentionally separate IDEA/AWM dependency environments.
RUN /opt/idea/bin/python - <<'PY'
from pathlib import Path
import subprocess
import tempfile
for name in ("idea", "awm"):
    python = "/opt/" + name + "/bin/python"
    requirements = Path("/opt/idea-runtime/" + name + "-requirements.txt").read_text().splitlines()
    requirements = [line for line in requirements if line.strip() and not line.startswith("#")]
    package_name = lambda line: line.split("==")[0].lower().replace("_", "-")
    bootstrap = [line for line in requirements if package_name(line) in {"pip", "setuptools", "wheel"}]
    flash = [line for line in requirements if package_name(line) == "flash-attn"]
    ordinary = [line for line in requirements if line not in flash]
    subprocess.run([python, "-m", "pip", "install", "--no-deps", *bootstrap], check=True)
    with tempfile.TemporaryDirectory(prefix="idea-image-pins-") as temporary:
        path = Path(temporary) / "requirements.txt"
        path.write_text("\n".join(ordinary) + "\n")
        subprocess.run([python, "-m", "pip", "install", "--no-deps", "-r", str(path)], check=True)
    if flash:
        subprocess.run([python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", *flash], check=True)
PY

# GPU-free version/import checks only. Workspaces supply IDEA/AWM source at run time.
RUN /opt/idea/bin/python -c "import torch, ray, transformers, flash_attn; import importlib.metadata as m; assert torch.__version__.split('+')[0]=='2.8.0'; assert torch.version.cuda=='12.8'; assert m.version('vllm')=='0.11.0'; assert m.version('flash-attn')=='2.7.4.post1'" && \
    /opt/awm/bin/python -c "import fastapi, fastapi_mcp, mcp_agent; import importlib.metadata as m; assert m.version('fastapi')=='0.115.12'; assert m.version('fastapi-mcp')=='0.4.0'; assert m.version('mcp-agent')=='0.2.6'"

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
