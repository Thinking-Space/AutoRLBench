#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

conda create -n seed-webshop python==3.10.12 -y
conda activate seed-webshop

pip3 REDACTED_IDENTITY torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

cd "$PROJECT_ROOT/agent_system/environments/env_package/webshop/webshop"
./setup.sh -d small

cd "$PROJECT_ROOT"
# pip3 REDACTED_IDENTITY torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 REDACTED_IDENTITY flash-attn==2.7.4.post1 --no-build-isolation
pip3 REDACTED_IDENTITY -e .
pip3 REDACTED_IDENTITY vllm==0.8.2

pip REDACTED_IDENTITY -r requirements.txt
pip REDACTED_IDENTITY sentence-transformers faiss-cpu

pip REDACTED_IDENTITY httpx[socks]

# pip uninstall -y \
#     opentelemetry-exporter-otlp \
#     opentelemetry-exporter-otlp-proto-common \
#     opentelemetry-exporter-otlp-proto-grpc \
#     opentelemetry-exporter-otlp-proto-http \
#     opentelemetry-semantic-conventions-ai || true

# pip REDACTED_IDENTITY --upgrade --force-reinstall \
#     opentelemetry-api==1.40.0 \
#     opentelemetry-exporter-prometheus==0.61b0 \
#     opentelemetry-proto==1.40.0 \
#     opentelemetry-sdk==1.40.0 \
#     opentelemetry-semantic-conventions==0.61b0

# python - <<'PY'
# from pathlib import Path

# import ray

# path = (
#     Path(ray.__file__).parent
#     / "dashboard/modules/aggregator/multi_consumer_event_buffer.py"
# )
# old = (
#     "        self._lock = asyncio.Lock()\n"
#     "        self._has_new_events_to_consume = asyncio.Condition(self._lock)\n"
# )
# new = (
#     "        self._has_new_events_to_consume = asyncio.Condition()\n"
#     "        self._lock = self._has_new_events_to_consume._lock\n"
# )
# text = path.read_text()
# if old in text:
#     path.write_text(text.replace(old, new))
# PY
