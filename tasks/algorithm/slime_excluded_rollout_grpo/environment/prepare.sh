#!/usr/bin/env bash
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_ENVIRONMENT:?}"

# Qwen/Qwen3.5-0.8B at 2fc06364715b967f1860aea9cf38778875588b17.
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python3 "${AUTORLBENCH_ENVIRONMENT}/fetch_hf.py" \
    --repo Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
    --dest "${AUTORLBENCH_WORKSPACE}/assets/model"

python3 - "${AUTORLBENCH_WORKSPACE}/assets/model" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
checksums = {
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "model.safetensors-00001-of-00001.safetensors": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    "model.safetensors.index.json": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
}
for name, expected in checksums.items():
    with (root / name).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Public checkpoint checksum mismatch for {name}")
PY
