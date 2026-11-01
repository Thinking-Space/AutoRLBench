#!/usr/bin/env bash
# Prepare the inputs for the Search-R1 slime recipe:
#   1. the QA dataset  -> nq_hotpotqa_train/{train,test}.parquet
#   2. the retrieval corpus -> wiki-18 e5 FAISS index + wiki-18.jsonl
#
# It drives the data-preparation subset of the Search-R1 repo that ships next to
# this recipe (workspace/Search-R1). Nothing here trains a model.
#
# Usage:
#   bash prepare_data.sh                 # dataset + corpus
#   DATASET_ONLY=1 bash prepare_data.sh  # skip the ~130GB corpus download
#   CORPUS_ONLY=1  bash prepare_data.sh  # only the retrieval corpus
#
# Override any path via the environment variables below.
set -euo pipefail

# --- locations -------------------------------------------------------------
# Directory of this script (examples/search-r1), resolved so it works from any cwd.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The trimmed Search-R1 repo. Defaults to the copy staged in this workspace
# (workspace/Search-R1); at container runtime the recipe reads /root/Search-R1,
# so allow an override.
SEARCH_R1_DIR="${SEARCH_R1_DIR:-$(cd "$HERE/../../Search-R1" && pwd)}"

# Where prepared assets land. Keep them inside the repo's data/ dir so the paths
# match the recipe's --prompt-data (…/Search-R1/data/nq_hotpotqa_train/…).
DATA_DIR="${DATA_DIR:-$PWD}"
DATASET_DIR="${DATASET_DIR:-$DATA_DIR/nq_hotpotqa_train}"
CORPUS_DIR="${CORPUS_DIR:-$DATA_DIR/wiki-18}"

# Which QA sources to merge. Train mixes NQ+HotpotQA (the recipe default); the
# test merge can span more sets for evaluation.
TRAIN_SOURCES="${TRAIN_SOURCES:-nq,hotpotqa}"
TEST_SOURCES="${TEST_SOURCES:-nq,hotpotqa}"

DATASET_ONLY="${DATASET_ONLY:-0}"
CORPUS_ONLY="${CORPUS_ONLY:-0}"
# Re-download even if the target files already exist.
FORCE="${FORCE:-0}"

if [[ ! -d "$SEARCH_R1_DIR/scripts/data_process" ]]; then
  echo "error: Search-R1 data-prep scripts not found under: $SEARCH_R1_DIR" >&2
  echo "       set SEARCH_R1_DIR to the trimmed Search-R1 repo." >&2
  exit 1
fi

# The merge scripts import verl.utils.hdfs_io from the repo root.
export PYTHONPATH="$SEARCH_R1_DIR${PYTHONPATH:+:$PYTHONPATH}"

# --- 1) QA dataset ---------------------------------------------------------
prepare_dataset() {
  local train_parquet="$DATASET_DIR/train.parquet"
  local test_parquet="$DATASET_DIR/test.parquet"

  if [[ "$FORCE" != "1" && -f "$train_parquet" && -f "$test_parquet" ]]; then
    echo "==> QA parquet already present, skipping ($DATASET_DIR)"
    echo "    -> $train_parquet"
    echo "    -> $test_parquet"
    return
  fi

  echo "==> Building QA parquet into $DATASET_DIR"
  echo "    train sources: $TRAIN_SOURCES"
  echo "    test  sources: $TEST_SOURCES"
  mkdir -p "$DATASET_DIR"

  if [[ "$FORCE" != "1" && -f "$train_parquet" ]]; then
    echo "    train.parquet exists, skipping train merge"
  else
    python "$SEARCH_R1_DIR/scripts/data_process/qa_search_train_merge.py" \
      --local_dir "$DATASET_DIR" --data_sources "$TRAIN_SOURCES"
  fi

  if [[ "$FORCE" != "1" && -f "$test_parquet" ]]; then
    echo "    test.parquet exists, skipping test merge"
  else
    python "$SEARCH_R1_DIR/scripts/data_process/qa_search_test_merge.py" \
      --local_dir "$DATASET_DIR" --data_sources "$TEST_SOURCES"
  fi

  echo "    -> $train_parquet"
  echo "    -> $test_parquet"
}

# --- 2) retrieval corpus (wiki-18 e5) --------------------------------------
prepare_corpus() {
  local index="$CORPUS_DIR/e5_Flat.index"
  local jsonl="$CORPUS_DIR/wiki-18.jsonl"

  if [[ "$FORCE" != "1" && -f "$index" && -f "$jsonl" ]]; then
    echo "==> wiki-18 corpus already present, skipping ($CORPUS_DIR)"
    echo "    -> $index"
    echo "    -> $jsonl"
    return
  fi

  echo "==> Downloading wiki-18 e5 index + corpus into $CORPUS_DIR"
  echo "    NOTE: the assembled index is ~130GB; ensure disk + bandwidth."
  mkdir -p "$CORPUS_DIR"

  # Pulls PeterJinGo/wiki-18-e5-index (part_aa, part_ab) and
  # PeterJinGo/wiki-18-corpus (wiki-18.jsonl.gz) from the HuggingFace Hub.
  python "$SEARCH_R1_DIR/scripts/download.py" --save_path "$CORPUS_DIR"

  echo "    Assembling e5_Flat.index from shards"
  cat "$CORPUS_DIR"/part_* > "$index"
  rm -f "$CORPUS_DIR"/part_*

  if [[ -f "$CORPUS_DIR/wiki-18.jsonl.gz" ]]; then
    echo "    Decompressing wiki-18.jsonl"
    gzip -df "$CORPUS_DIR/wiki-18.jsonl.gz"
  fi

  echo "    -> $index"
  echo "    -> $jsonl"
}

# --- run -------------------------------------------------------------------
if [[ "$CORPUS_ONLY" == "1" ]]; then
  prepare_corpus
elif [[ "$DATASET_ONLY" == "1" ]]; then
  prepare_dataset
else
  prepare_dataset
  prepare_corpus
fi

echo "==> Done."
