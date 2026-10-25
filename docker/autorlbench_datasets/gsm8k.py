"""openai/gsm8k -> gsm8k_verl/{train,test}.parquet (used by task_engine_base_025)

**No custom split needed.** The historical output is train (7473, 5) / test (1319, 5),
which is exactly the train/test split shipped with upstream `openai/gsm8k` config `main`,
so we reuse the upstream split and invent no boundaries of our own.

The target schema comes from a real dump in a historical agent_trace (025's agent ran
`pd.read_parquet(...).head(3).to_dict('records')`):

    data_source   'openai/gsm8k'        <- reward_score/__init__.py:127 matches this string exactly
    prompt        [{'content': <question> + ' ' + INSTRUCTION, 'role': 'user'}]
    ability       'math'
    reward_model  {'ground_truth': '72', 'style': 'rule'}
    extra_info    {'answer': <full solution>, 'index': i, 'question': <original question>, 'split': 'train'}

That is the output of verl's own examples/data_preprocess/gsm8k.py (that directory was not
vendored into the task workspace, so it is rebuilt here).

Usage:
    python3 -m autorlbench_datasets.gsm8k --out-dir ./prepared/gsm8k_verl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ._common import hf_datasets_cache, report, write_split

REPO = "openai/gsm8k"
CONFIG = "main"
DATA_SOURCE = "openai/gsm8k"  # must match character for character, or the scorer will not route to gsm8k

# Instruction suffix appended by verl's stock gsm8k preprocessing; matches the historical output.
INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'

# Row counts of the historical output. These are upstream main's official splits; a mismatch
# means upstream changed, so stop.
EXPECTED = {"train": 7473, "test": 1319}

# Part of the cache key; bump it whenever the recipe changes. Same reason as dapo_math_17k.RECIPE_VERSION.
RECIPE_VERSION = "1"

_SOLUTION_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_ground_truth(answer: str) -> str:
    """gsm8k answers end with `#### <number>`; the ground truth is that number (commas stripped)."""
    m = _SOLUTION_RE.search(answer)
    if m is None:
        raise ValueError(f"no '#### <number>' in answer: {answer[-120:]!r}")
    return m.group(1).replace(",", "")


def build_rows(split_name: str, ds) -> list[dict]:
    rows = []
    for idx, item in enumerate(ds):
        question_raw = item["question"]
        answer_raw = item["answer"]
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"content": f"{question_raw} {INSTRUCTION}", "role": "user"}],
                "ability": "math",
                "reward_model": {
                    "ground_truth": extract_ground_truth(answer_raw),
                    "style": "rule",
                },
                "extra_info": {
                    "answer": answer_raw,
                    "index": idx,
                    "question": question_raw,
                    "split": split_name,
                },
            }
        )
    return rows


def build(out_dir: Path, *, allow_count_mismatch: bool = False) -> list[Path]:
    """Download upstream, rebuild the verl schema, write both parquets; return the written paths.

    Kept separate from main() for the same reason as dapo_math_17k.build(): the in-container
    `autorlbench_dataset` calls it inside the cache directory, then hardlinks the output to
    where the task expects it.
    """
    # Before the import: see _common.hf_cache_root(). Without cache_dir, upstream lands in
    # ~/.cache/huggingface, which is the container's writable layer: nothing shows up in
    # the host's datasets/, and every recipe rebuild downloads it again.
    cache_dir = hf_datasets_cache()
    from datasets import load_dataset

    print(f"downloading {REPO} (config={CONFIG}) ...")
    dataset = load_dataset(REPO, CONFIG, cache_dir=str(cache_dir))

    written = []
    for split_name in ("train", "test"):
        ds = dataset[split_name]
        want = EXPECTED[split_name]
        if len(ds) != want:
            msg = f"{split_name}: upstream has {len(ds)} rows, historical output has {want}"
            if not allow_count_mismatch:
                raise SystemExit(
                    f"{msg}. Upstream data changed; the grader's baseline may no longer be comparable.\n"
                    f"Once confirmed, rerun with --allow-count-mismatch."
                )
            print(f"!! {msg} (--allow-count-mismatch is set, continuing)")

        rows = build_rows(split_name, ds)
        path = write_split(rows, out_dir, split_name)
        print(f"  {split_name}: {len(rows)} rows -> {path}")
        written.append(path)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="continue even if upstream row counts differ from the historical output (fails by default)",
    )
    args = ap.parse_args()

    report(build(args.out_dir, allow_count_mismatch=args.allow_count_mismatch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
