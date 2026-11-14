#!/usr/bin/env bash
# verl_taco_wrong_expected_output — prepare stage.
#
# Runs in a separate prepare-stage container: root, the verl image, the only
# stage with network, and $AUTORLBENCH_WORKSPACE / $AUTORLBENCH_PREPARED /
# $AUTORLBENCH_MODELS / $AUTORLBENCH_DATASETS all writable.
#
# In the agent stage this file is mounted read-only at /autorlbench/environment
# and the agent can read it: maintainer notes (calibration status, evidence
# plan, hardening TODOs) go in ../README.md, not here.
#
# WHAT THIS TASK FETCHES
#
#   Qwen/Qwen3-4B-Instruct-2507  model (non-thinking instruct model: the
#                                taco.py reward's post_process takes the first
#                                fenced code block, so a thinking template
#                                would score scratchpad code)
#   BAAI/TACO                    ALL/train-* + ALL/test shards only (--allow):
#                                the native, unfiltered training pool and the
#                                held-out split
#   likaixin/TACO-verified       taco_verified.json: the execution-verified
#                                subset (12,898 of 25,443 train problems)
#
# and assembles $AUTORLBENCH_WORKSPACE/taco_verl/{train,test}.parquet from them
# with assemble_dataset.py plus the committed selection inputs in selection/.
# The assembly itself is deterministic (seeded); see assemble_dataset.py.
#
# The repos are fetched at pinned revisions (--revision, see fetch_hf.py's
# docstring): the committed selection inputs are only valid against the
# revision they were purified from. Byte sizes, SHA-256 values and the revision
# SHAs are recorded in ../README.md for reference (fetch_hf.py is cache-first,
# no checksum verification).
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_ENVIRONMENT:?}"

MODEL_DEST="$AUTORLBENCH_WORKSPACE/Qwen3-4B-Instruct-2507"

# --- Base model -------------------------------------------------------------
# In examples/grpo_trainer/run_qwen3_4b_instruct_2507_taco_fsdp.sh:22,
#   MODEL_PATH="${MODEL_PATH:-${VERL_ROOT}/Qwen3-4B-Instruct-2507}"
# VERL_ROOT is two levels above the launch script, i.e. the workspace root in
# the container, so dest must be exactly $AUTORLBENCH_WORKSPACE/Qwen3-4B-Instruct-2507.
# revision cdbee75f17c01a7cc42f958dc650907174af0554 (main, 2026-11-09).
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen3-4B-Instruct-2507 \
    --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
    --dest "$MODEL_DEST"

# --- Sources ----------------------------------------------------------------
# Staged under $AUTORLBENCH_PREPARED (NOT the workspace): the assembled
# taco_verl parquets are what the run needs; keeping the 4+ GB of raw sources
# out of the workspace avoids copying them into every agent/verifier stage.
# The shared download cache still holds the bytes across runs (cache-first),
# so a rerun pays nothing but the hardlinks.
SOURCES="$AUTORLBENCH_PREPARED/taco-sources"
rm -rf "$SOURCES"
mkdir -p "$SOURCES"

python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --type dataset \
    --repo BAAI/TACO \
    --revision d593ed0a2becbbc952230bb89be09189bf1056dc \
    --allow 'ALL/train-*-of-00009.parquet' \
    --allow 'ALL/test-*-of-*.parquet' \
    --dest "$SOURCES/BAAI__TACO"

python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --type dataset \
    --repo likaixin/TACO-verified \
    --revision 9e68575e01938f016199eda6f07d1a7ce96b35e1 \
    --dest "$SOURCES/likaixin__TACO-verified"

# assemble_dataset.py expects the shards and taco_verified.json side by side.
for shard in "$SOURCES"/BAAI__TACO/ALL/train-*-of-00009.parquet \
             "$SOURCES"/BAAI__TACO/ALL/test-*-of-*.parquet; do
    ln -f "$shard" "$SOURCES/$(basename "$shard")"
done
ln -f "$SOURCES"/likaixin__TACO-verified/taco_verified.json "$SOURCES/taco_verified.json"

# --- Assembly ---------------------------------------------------------------
echo "[prepare] assembling taco_verl train/test parquets"
rm -rf "$AUTORLBENCH_WORKSPACE/taco_verl"
python3 "$AUTORLBENCH_ENVIRONMENT/assemble_dataset.py" \
    --sources "$SOURCES" \
    --manifest "$AUTORLBENCH_ENVIRONMENT/selection/dirty_manifest.jsonl" \
    --test-manifest "$AUTORLBENCH_ENVIRONMENT/selection/test_manifest.jsonl" \
    --mix-spec "$AUTORLBENCH_ENVIRONMENT/selection/mix_spec.json" \
    --tokenizer "$MODEL_DEST" \
    --out "$AUTORLBENCH_WORKSPACE/taco_verl"

# The raw sources are no longer needed once the parquets exist; the shared
# cache keeps them for the next prepare anyway.
rm -rf "$SOURCES"
