#!/usr/bin/env bash
# Recorded assets are immutable. Every downloaded file is size+SHA checked
# against assets.json before use; prepare_data.py re-verifies the selected rows
# and the produced parquet outputs.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONDONTWRITEBYTECODE=1

# prepare_data.py reads/writes parquet and pins outputs by content digest, so
# any pyarrow that round-trips the recorded content passes; only install if the
# image is missing pyarrow entirely.
python3 -c 'import pyarrow' 2>/dev/null \
  || uv pip install --system --python "$(command -v python3)" \
       'pyarrow==21.0.0'

# Pinned assets are fetched with the standard autorlbench_fetch_hf tool,
# driven by the revision + per-file size/SHA pins in assets.json, into the
# shared cache roots. prepare_data.py then rebuilds the
# selected rows and re-verifies the produced parquet outputs.
python3 - "$HERE/assets.json" "$AUTORLBENCH_WORKSPACE" <<'EOF'
import json, subprocess, sys
from pathlib import Path
pins = json.loads(Path(sys.argv[1]).read_text())
resources = Path(sys.argv[2]) / "resources"
for a in pins:
    argv = ["autorlbench_fetch_hf",
            "--repo", a["repository"], "--type", a["kind"], "--revision", a["revision"]]
    for f in a["files"]:
        if f["path"] == ".gitattributes":
            continue
        argv += ["--allow", f["path"],
                 "--expect", f"{f['path']}={f['sha256']}",
                 "--expect-size", f"{f['path']}={f['bytes']}"]
    dest = resources / ("model" if a["kind"] == "model" else "raw")
    argv += ["--dest", str(dest)]
    print("[prepare] " + " ".join(argv), flush=True)
    subprocess.run(argv, check=True)
EOF

python3 "$AUTORLBENCH_ENVIRONMENT/prepare_data.py" \
  --raw-root "$AUTORLBENCH_WORKSPACE/resources/raw" \
  --output "$AUTORLBENCH_WORKSPACE/resources"
