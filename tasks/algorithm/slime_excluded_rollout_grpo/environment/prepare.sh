#!/usr/bin/env bash
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
python3 - "$AUTORLBENCH_WORKSPACE" <<'PY'
import json
import sys
from pathlib import Path

import torch

workspace = Path(sys.argv[1])
assert torch.tensor([1.0]).device.type == "cpu"
assert len(json.loads((workspace / "batches.json").read_text())) == 6
assert (workspace / "evidence/recorded-replay.json").is_file()
PY
