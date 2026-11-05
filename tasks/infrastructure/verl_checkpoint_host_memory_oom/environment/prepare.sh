#!/usr/bin/env bash
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_ENVIRONMENT:?}"
# Complete official GSM8K train/test splits ship with this task.
# environment/data_manifest.json records the pinned upstream files and row selection.
# The task-local fetch_hf.py fetches the whole repo, cache-first, without
# checksum verification (data_manifest.json is the pin of record); the
# revision the former autorlbench_fetch_hf allowlist pinned, for reference:
#   Qwen/Qwen2.5-0.5B-Instruct @ 7ae557604adf67be50417f59c2c2f167def9a775
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
  --repo Qwen/Qwen2.5-0.5B-Instruct \
  --dest "$AUTORLBENCH_WORKSPACE/model"
