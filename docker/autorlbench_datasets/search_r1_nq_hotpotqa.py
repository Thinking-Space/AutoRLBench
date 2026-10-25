"""RUC-NLPIR/FlashRAG_datasets -> nq_hotpotqa_train/{train,test}.parquet

Shared by tasks/slime_search_r1 and tasks/slime_search_r1_nomask (both tasks' launch
scripts want /workspace/nq_hotpotqa_train/{train,test}.parquet; see
examples/search-r1/run_qwen2.5_3B.sh:64-66).

## What this recipe is

An equivalent of the two merge scripts that ship with the recipe:

    workspace/Search-R1/scripts/data_process/qa_search_train_merge.py
    workspace/Search-R1/scripts/data_process/qa_search_test_merge.py

Those scripts call `datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', <source>)`,
which **does not pin a revision** and only works in a process with network. prepare is the
only stage with network, but the output must be reusable across runs (each evaluation gets a
fresh workspace copy), so it is rebuilt here following the autorlbench_datasets conventions:
pinned revision, asserted row counts, parquet written via pandas.

The prompt template, field names, and how `extra_info.index` counts are copied **verbatim**
from those two scripts (see the make_prefix / build_rows comments below). Changing this means
changing them too, and vice versa: if the outputs diverge, an agent that reruns prepare_data.sh
in /workspace gets data that differs from the grading baseline.

## The only difference from those scripts: no metadata column

Each upstream hotpotqa row carries a `metadata` field (supporting_facts + the full text of ten
context passages, most of the 569MB), while nq rows have no such key at all. The merge scripts
absorb this via `datasets` schema alignment; with pandas, a column that is "half struct, half
missing" would need extra null-filling logic and would haul 500MB the training side never reads
into every run's workspace.

slime only reads `--input-key prompt` and `--label-key reward_model`
(ROLLOUT_ARGS in run_qwen2.5_3B.sh), and the reward only uses `reward_model.ground_truth.target`
(examples/search-r1/qa_em_format.py), so dropping metadata changes no training signal or
grading input. id / question / golden_answers are kept because both sources have them and they
help when dumping rows to debug.

## Row counts

    train = nq/train.jsonl 79168 + hotpotqa/train.jsonl 90447 = 169615
    test  = nq/test.jsonl   3610 + hotpotqa/dev.jsonl    7405 =  11015

The order **must not change**: the headline metric is `eval/nq_test`, and the recipe uses
`--eval-prompt-data nq_test "$TEST_DATA@[0:3000]"` (ROLLOUT_ARGS in run_qwen2.5_3B.sh),
which relies on the first 3610 rows of test.parquet all being NQ. With hotpotqa first, the
metric would still be named nq_test but evaluate a different set, with no error.

hotpotqa's test set comes from dev.jsonl because upstream hotpotqa has no test split; the merge
script's test / dev / train priority is in qa_search_test_merge.py:56-63 and is mirrored here.

Usage:
    python3 -m autorlbench_datasets.search_r1_nq_hotpotqa --out-dir ./prepared/nq_hotpotqa_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._common import hf_cache_root, report, write_split

REPO = "RUC-NLPIR/FlashRAG_datasets"
# Pin the revision: upstream is a 91-file collection repo that may add or change a dataset at any time.
# The grading baseline was measured against this one.
REVISION = "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"

# This order is the row order of the output; see the module docstring. The merge scripts default
# to `nq,hotpotqa` (TRAIN_SOURCES / TEST_SOURCES in prepare_data.sh:34-35).
SOURCES = ("nq", "hotpotqa")

# The configs block in the upstream README maps each config to these jsonl files, so fetching
# the files directly reads the same bytes as load_dataset(REPO, config).
FILES = {
    "train": {"nq": "nq/train.jsonl", "hotpotqa": "hotpotqa/train.jsonl"},
    "test": {"nq": "nq/test.jsonl", "hotpotqa": "hotpotqa/dev.jsonl"},
}

# Row counts per source and split. A mismatch means upstream changed, so stop:
# required_final_score was measured on this data.
EXPECTED = {
    "train": {"nq": 79168, "hotpotqa": 90447},
    "test": {"nq": 3610, "hotpotqa": 7405},
}

# Part of the cache key (see docker/autorlbench_dataset.py). **Bump this whenever the recipe
# changes**: if the prompt template, column selection, or row order change, the previously cached
# parquet is no longer this recipe's output, and whichever task runs first keeps getting stale bytes.
RECIPE_VERSION = "1"


def make_prefix(question: str) -> str:
    """Copied verbatim from make_prefix(template_type='base') in qa_search_{train,test}_merge.py.

    Every space in the template went into the historical runs' prompts, and the reward's format
    state machine (qa_em_format.py) expects the <think>/<search>/<information>/<answer> tags, so
    this text is part of the data, not commentary.
    """
    return (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results between <information> "
        "and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly provide the answer inside "
        "<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
        f"Question: {question}\n"
    )


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_rows(source: str, split_name: str, raw: list[dict]) -> list[dict]:
    """Turn one source's raw rows into verl-style rows.

    `index` counts from 0 **within each source**, not globally after merging: the merge scripts
    run `.map(with_indices=True)` per source and then concatenate, so hotpotqa's index restarts
    at 0. Copied not because it is better, but because the output must match those scripts.
    """
    rows = []
    for idx, item in enumerate(raw):
        question = item["question"].strip()
        if not question:
            raise SystemExit(f"{source}/{split_name} row {idx} has an empty question")
        # The merge scripts modify example['question'] in place before templating, so the
        # edited question stays in the output.
        if question[-1] != "?":
            question += "?"
        golden = item.get("golden_answers")
        if golden is None:
            raise SystemExit(
                f"{source}/{split_name} row {idx} has no golden_answers, "
                f"which is the key reward_model.ground_truth.target reads"
            )
        rows.append(
            {
                "id": item.get("id"),
                "question": question,
                "golden_answers": golden,
                "data_source": source,
                "prompt": [{"role": "user", "content": make_prefix(question)}],
                "ability": "fact-reasoning",
                "reward_model": {"style": "rule", "ground_truth": {"target": golden}},
                "extra_info": {"split": split_name, "index": idx},
            }
        )
    return rows


def build(out_dir: Path, *, allow_count_mismatch: bool = False) -> list[Path]:
    """Download the upstream jsonl files and rebuild both parquets; return the written paths.

    Kept separate from main() for the same reason as dapo_math_17k.build(): the in-container
    `autorlbench_dataset` calls it inside the cache directory, then hardlinks the output to
    where the task expects it.
    """
    # Before the import: see _common.hf_cache_root(). The jsonl files themselves land on the host
    # via local_dir below; this covers huggingface_hub's own metadata/etag cache, which would
    # otherwise go to the container's writable layer.
    hf_cache_root()
    from huggingface_hub import hf_hub_download

    # Raw jsonl goes under out_dir and is then deleted: `local_dir` makes hf_hub_download write
    # plain files without keeping a second blob cache elsewhere. Cache entries take effect by
    # rename, so a tree left behind by a failure midway is discarded whole by autorlbench_dataset.
    raw_dir = out_dir / "_raw"
    written = []
    for split_name in ("train", "test"):
        rows: list[dict] = []
        for source in SOURCES:
            filename = FILES[split_name][source]
            print(f"downloading {REPO}@{REVISION[:12]} {filename} ...")
            path = Path(
                hf_hub_download(
                    repo_id=REPO,
                    filename=filename,
                    repo_type="dataset",
                    revision=REVISION,
                    local_dir=str(raw_dir),
                )
            )
            raw = read_jsonl(path)
            want = EXPECTED[split_name][source]
            if len(raw) != want:
                message = f"{filename} has {len(raw)} rows, the recipe expects {want}"
                if not allow_count_mismatch:
                    raise SystemExit(
                        f"{message}. Upstream changed; once confirmed, add --allow-count-mismatch."
                    )
                print(f"!! {message} (--allow-count-mismatch is set, continuing)")
            rows.extend(build_rows(source, split_name, raw))
            # Delete once read: hotpotqa/train.jsonl alone is 569MB, and keeping it in the cache
            # entry means every run's workspace hauls bytes nobody reads.
            path.unlink()
        path = write_split(rows, out_dir, split_name)
        print(f"  {split_name}: {len(rows)} rows -> {path}")
        written.append(path)

    # Only hf_hub_download's bookkeeping directory (.cache/huggingface) is left under `_raw`.
    # Left here, it would be hardlinked into the workspace along with the output.
    if raw_dir.exists():
        import shutil

        shutil.rmtree(raw_dir)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--allow-count-mismatch", action="store_true")
    args = ap.parse_args()

    written = build(args.out_dir, allow_count_mismatch=args.allow_count_mismatch)
    report(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
