#!/usr/bin/env bash
# search_r1_zero_query_vector: what to prepare before running.
#
# Runs in a separate prepare-stage container: root, `search-r1-verl` image, **the only stage with network**,
# and $AUTORLBENCH_WORKSPACE, $AUTORLBENCH_MODELS, $AUTORLBENCH_DATASETS are all writable.
# Network cutoff for the agent / verifier stages is set per stage by the framework (HF_HUB_OFFLINE /
# TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE in _network_policy_env in runtimes/base.py). Do **not**
# repeat it in the task's metadata.environment.env: that env applies to all three stages alike and
# would also cut the network prepare itself needs.
#
# Three things: two models (about 3.5GB) are downloaded into the workspace, and one FAISS index (37MB)
# is rebuilt in place. The data itself (train/test.parquet + corpus.jsonl, 6.9MB) ships with the task
# package and is not downloaded here. Target directory names come from the SHARED constants in
# workspace/run.py and workspace/serve_retriever.py:
#
#   shared/models/Qwen2.5-1.5B-Instruct   initial actor / ref weights (BASE_MODEL)
#   shared/models/e5-base-v2              query/document encoder for retrieval
#   shared/data/{train,test}.parquet      shipped with the package
#   shared/data/corpus.jsonl              shipped with the package
#   shared/data/e5_Flat.index             rebuilt here
#
# Before starting, run.py checks each of these five paths (its `required` list) and refuses to run if
# any is missing, so a wrong path fails up front instead of halfway through training.
set -euo pipefail

ENVIRONMENT_DIR="${AUTORLBENCH_ENVIRONMENT:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
DATA_DIR="$AUTORLBENCH_WORKSPACE/shared/data"
MODEL_DIR="$AUTORLBENCH_WORKSPACE/shared/models"

# ---------------------------------------------------------------------------
# Integrity of the shipped data
# ---------------------------------------------------------------------------
# These three files live in git and are not downloaded, so this checks the checkout, not a download:
# if LFS is not set up or someone edited a parquet by hand, the symptom would otherwise be a wrong row
# count at the first eval. It also ties data_manifest.json to the actual files. The manifest is the
# provenance cited by the readme and oracle, so if it drifts, nothing lines up.
echo "[prepare] verifying the data shipped in the task package"
python3 - "$ENVIRONMENT_DIR/data_manifest.json" "$DATA_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, data_dir = Path(sys.argv[1]), Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for relative, expected in manifest["shipped_in_workspace"].items():
    path = data_dir.parent.parent / relative
    if not path.is_file():
        raise SystemExit(f"[prepare] missing shipped data file: {path}")
    size = path.stat().st_size
    if size != expected["size"]:
        raise SystemExit(
            f"[prepare] {path}: {size} bytes, manifest says {expected['size']}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected["sha256"]:
        raise SystemExit(
            f"[prepare] {path}: sha256 {digest}, manifest says {expected['sha256']}"
        )
    print(f"[prepare]   ok {relative}")
PY

# ---------------------------------------------------------------------------
# Initial actor / ref weights
# ---------------------------------------------------------------------------
# The revision is the commit used by the experiment recorded in the delivery package README; the sha256
# comes from MODEL_SHA256SUMS and matches input_sha256 in results/baseline32/experiment.json
# (that entry lists model.safetensors as dd924a11...). So the weights downloaded here are the same
# file that produced the 15.6% curve.
#
# The allow list excludes .gitattributes and README; the remaining config / tokenizer files are read by
# transformers itself. Only model.safetensors gets a sha256: the revision already pins the small files,
# and listing each one would just make the task package harder to write.
autorlbench_fetch_hf \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    --allow config.json \
    --allow generation_config.json \
    --allow merges.txt \
    --allow model.safetensors \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.json \
    --expect model.safetensors=dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee \
    --dest "$MODEL_DIR/Qwen2.5-1.5B-Instruct"

# ---------------------------------------------------------------------------
# Retrieval encoder
# ---------------------------------------------------------------------------
# The delivery package did not record the e5 commit (the README says so), but the model.safetensors
# downloaded at revision f52bf8ec... has a sha256 byte-identical to the one in the delivery package
# (d0d559c4...), and tasks/slime_search_r1_nomask already pins the same file at this revision, so the
# two tasks share one cache entry. If the sha does not match, stop and check the version; do not pair
# this index with a different encoder version. Document vectors and query vectors must come from the
# same weights, or retrieval quality degrades silently.
#
# The onnx/ and openvino/ subtrees are skipped: they are the same weights in two other formats, and
# nothing here reads them.
autorlbench_fetch_hf \
    --repo intfloat/e5-base-v2 \
    --revision f52bf8ec8c7124536f0efb74aca902b2995e5bcd \
    --allow config.json \
    --allow model.safetensors \
    --allow special_tokens_map.json \
    --allow tokenizer.json \
    --allow tokenizer_config.json \
    --allow vocab.txt \
    --expect model.safetensors=d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693 \
    --expect-size model.safetensors=437955512 \
    --dest "$MODEL_DIR/e5-base-v2"

# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------
# Why it is not shipped: it can be computed from corpus.jsonl plus a pinned encoder, and at 37MB it is
# five times the size of its own source file. Why the agent stage does not build it: that stage is
# timed, and this step needs no agent decisions and can be reused across runs.
#
# The real file lives under $AUTORLBENCH_DATASETS and survives across runs, with a hard link into the
# workspace; on the same filesystem that copy costs no bytes (like other tasks' materialize), and across
# filesystems it falls back to a real copy. From the second run on, the few minutes of encoding are skipped.
#
# A .complete marker is used instead of trusting that the index file exists: write_index replaces the
# file atomically in build_index.py, but the cache directory may be a half-finished state left by a
# previously killed prepare, and trusting file existence would treat it as built.
# $AUTORLBENCH_DATASETS is the empty string when the harness has no dataset cache
# configured (the local backend can run that way). Falling through with an empty
# value would build the cache at /search_r1_hotpotqa_e5; keep it in the run's own
# workspace instead, where it is at least correct, just not reused.
INDEX_CACHE="${AUTORLBENCH_DATASETS:-$AUTORLBENCH_WORKSPACE/.index-cache}/search_r1_hotpotqa_e5"
INDEX_CACHED="$INDEX_CACHE/e5_Flat.index"
INDEX_MARKER="$INDEX_CACHE/.complete"

if [ ! -f "$INDEX_MARKER" ] || [ ! -s "$INDEX_CACHED" ]; then
    rm -rf "$INDEX_CACHE"
    mkdir -p "$INDEX_CACHE"
    echo "[prepare] building $INDEX_CACHED from corpus.jsonl"
    python3 "$ENVIRONMENT_DIR/build_index.py" \
        --corpus "$DATA_DIR/corpus.jsonl" \
        --model "$MODEL_DIR/e5-base-v2" \
        --out "$INDEX_CACHED"
    touch "$INDEX_MARKER"
else
    echo "[prepare] reusing the cached index at $INDEX_CACHED"
fi

rm -f "$DATA_DIR/e5_Flat.index"
cp -al "$INDEX_CACHED" "$DATA_DIR/e5_Flat.index" 2>/dev/null \
    || cp -a "$INDEX_CACHED" "$DATA_DIR/e5_Flat.index"

# The retrieval service is not started here: prepare is a one-shot container that is torn down once done
# (_run_prepare in evaluation/evaluator.py), so its processes do not survive into the agent stage. This
# stage is also awaited in the foreground, so starting the service without detaching would just hang until
# timeout. That job belongs to workspace/run.py: it launches serve_retriever.py in its own foreground
# process, waits for /openapi.json to be ready, and shuts it down at the end.
#
# The corpus arrow cache is not prewarmed here either (slime_search_r1_nomask's prepare does that):
# that corpus is 14GB and takes over ten minutes to convert from cold; this one is 6.6MB and the
# datasets conversion takes seconds.
echo "[prepare] assets staged under $AUTORLBENCH_WORKSPACE/shared"
echo "[prepare] next step is the agent's: python3 run.py --check validates them without a GPU"
