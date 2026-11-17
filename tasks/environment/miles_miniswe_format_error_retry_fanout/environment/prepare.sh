#!/usr/bin/env bash
# Prepare the public Miles + Harbor assets used by this task.
set -euo pipefail

: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_PREPARED:?}"
WORKSPACE="$AUTORLBENCH_WORKSPACE"
mkdir -p "$AUTORLBENCH_PREPARED/src" "$AUTORLBENCH_PREPARED/models"

MILES_REPOSITORY="${MILES_REPOSITORY:-https://github.com/radixark/miles.git}"
MILES_COMMIT="${MILES_COMMIT:-64772f50b20b6e39051922a4033f3cb6019985dc}"
HARBOR_REPOSITORY="${HARBOR_REPOSITORY:-https://github.com/harbor-framework/harbor.git}"
HARBOR_COMMIT="${HARBOR_COMMIT:-0c2e1953c638f392ebbcf31d04883cb5e16edb22}"
HARBOR_INSTALL_REPOSITORY="${HARBOR_INSTALL_REPOSITORY:-git+${HARBOR_REPOSITORY}@${HARBOR_COMMIT}}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

MILES_SOURCE_DIR="${MILES_SOURCE_DIR:-$WORKSPACE/miles}"
HARBOR_SOURCE_DIR="${HARBOR_SOURCE_DIR:-$AUTORLBENCH_PREPARED/src/harbor}"

if [ ! -d "$MILES_SOURCE_DIR" ]; then
  git clone "$MILES_REPOSITORY" "$MILES_SOURCE_DIR"
fi
if [ -e "$MILES_SOURCE_DIR/.git" ]; then
  if ! git -C "$MILES_SOURCE_DIR" cat-file -e "$MILES_COMMIT^{commit}" 2>/dev/null; then
    git -C "$MILES_SOURCE_DIR" fetch origin "$MILES_COMMIT" --depth=1
  fi
  git -C "$MILES_SOURCE_DIR" checkout --quiet --force "$MILES_COMMIT"
elif [ ! -f "$MILES_SOURCE_DIR/miles/__init__.py" ]; then
  echo "[prepare] vendored Miles source is incomplete: $MILES_SOURCE_DIR" >&2
  exit 1
fi

if [ ! -e "$HARBOR_SOURCE_DIR/.git" ]; then
  git clone "$HARBOR_REPOSITORY" "$HARBOR_SOURCE_DIR"
fi
if ! git -C "$HARBOR_SOURCE_DIR" cat-file -e "$HARBOR_COMMIT^{commit}" 2>/dev/null; then
  git -C "$HARBOR_SOURCE_DIR" fetch origin "$HARBOR_COMMIT" --depth=1
fi
git -C "$HARBOR_SOURCE_DIR" checkout --quiet --force "$HARBOR_COMMIT"

if ! command -v uv >/dev/null 2>&1; then
  echo "[prepare] uv is required to install the Harbor branch" >&2
  exit 1
fi
uv pip install --python "$(command -v python3)" \
  "harbor[e2b] @ ${HARBOR_INSTALL_REPOSITORY}" \
  "e2b==2.50.0"

python3 - <<'PY'
import harbor
from harbor.models.environment_type import EnvironmentType

assert EnvironmentType.E2B.value == "e2b"
print(f"[prepare] harbor import OK: {harbor.__file__}")
import e2b
import importlib.metadata
assert importlib.metadata.version("e2b") == "2.50.0", importlib.metadata.version("e2b")
print(f"[prepare] e2b import OK: {e2b.__file__}")
PY

PYTHONPATH="$WORKSPACE" python3 - <<'PY'
import tencent_ags_e2b

assert tencent_ags_e2b.E2BEnvironment.__name__ == "E2BEnvironment"
print("[prepare] Tencent AGS Harbor environment import OK")
PY

if [ ! -d "$MEGATRON_PATH" ]; then
  echo "[prepare] Megatron-LM not found at $MEGATRON_PATH" >&2
  echo "[prepare] run this task in the shared Miles image, or set MEGATRON_PATH" >&2
  exit 1
fi

for required in \
  data/swebench_verified.jsonl \
  data/miles_prompts.jsonl \
  data/selected_instance_ids.txt \
  harbor_tasks/image_manifest.json; do
  if [ ! -s "$WORKSPACE/$required" ]; then
    echo "[prepare] required shipped asset is missing or empty: $WORKSPACE/$required" >&2
    exit 1
  fi
done

python3 - "$WORKSPACE" <<'PY'
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
ids = (workspace / "data/selected_instance_ids.txt").read_text().splitlines()
prompts = [
    json.loads(line)
    for line in (workspace / "data/miles_prompts.jsonl").read_text().splitlines()
    if line.strip()
]
records = [
    json.loads(line)
    for line in (workspace / "data/swebench_verified.jsonl").read_text().splitlines()
    if line.strip()
]

assert len(ids) == 16, f"expected 16 selected IDs, got {len(ids)}"
assert len(prompts) == 16, f"expected 16 prompts, got {len(prompts)}"
assert len(records) == 16, f"expected 16 records, got {len(records)}"
assert [row["metadata"]["instance_id"] for row in prompts] == ids
assert [row["instance_id"] for row in records] == ids

for instance_id in ids:
    task_dir = workspace / "harbor_tasks" / instance_id
    for relative in (
        "instruction.md",
        "task.toml",
        "environment/Dockerfile",
        "tests/config.json",
        "tests/test.sh",
        "solution/solve.sh",
    ):
        assert (task_dir / relative).is_file(), f"missing {task_dir / relative}"

import json
import re
import tomllib

manifest_path = workspace / "harbor_tasks/image_manifest.json"
assert manifest_path.is_file(), f"missing {manifest_path}"
manifest = {row["instance_id"]: row for row in json.loads(manifest_path.read_text())}
assert set(manifest) == set(ids), "image manifest does not match selected task IDs"

for instance_id in ids:
    task_dir = workspace / "harbor_tasks" / instance_id
    with (task_dir / "task.toml").open("rb") as task_file:
        config = tomllib.load(task_file)
    image = config["environment"].get("docker_image")
    assert image, f"missing [environment].docker_image for {instance_id}"
    assert manifest[instance_id]["docker_image"] == image
    dockerfile = (task_dir / "environment/Dockerfile").read_text(encoding="utf-8")
    match = re.search(r"^FROM\s+(\S+)", dockerfile, re.MULTILINE)
    assert match, f"missing Dockerfile FROM for {instance_id}"
    assert match.group(1) == image, (
        f"task.toml and Dockerfile images differ for {instance_id}: "
        f"{image!r} != {match.group(1)!r}"
    )

print(f"[prepare] validated {len(ids)} Harbor tasks")
PY

python3 -m py_compile \
  "$WORKSPACE/generate_harbor_tasks.py" \
  "$WORKSPACE/run_miles.py" \
  "$WORKSPACE/smoke_tencent_ags.py" \
  "$WORKSPACE/summarize_run.py" \
  "$WORKSPACE/tencent_ags_e2b.py" \
  "$WORKSPACE/tencent_harbor_agent.py"

cat > "$AUTORLBENCH_PREPARED/env.miles-miniswe-harbor" <<EOF
MILES_SOURCE_DIR=$MILES_SOURCE_DIR
HARBOR_SOURCE_DIR=$HARBOR_SOURCE_DIR
MEGATRON_PATH=$MEGATRON_PATH
MILES_MODEL_BASE=$AUTORLBENCH_PREPARED/models
EOF
while IFS= read -r line; do
  if ! grep -qs "^${line%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
    echo "$line" >> "${AUTORLBENCH_PREPARED}/env"
  fi
done < "$AUTORLBENCH_PREPARED/env.miles-miniswe-harbor"

echo "[prepare] public Miles and Harbor assets are ready"
