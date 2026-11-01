"""Materialize VERL-formatted GSM8K parquets from the pinned raw files.

Replicates verl's examples/data_preprocess/gsm8k.py: every row keeps exactly the
five columns RLHFDataset reads (data_source, prompt, ability, reward_model,
extra_info) with the same instruction suffix and strict "####" extraction.

The outputs are pinned by CONTENT, not by parquet bytes: the parquet writer
varies with the pyarrow version on the image, while the row content does not.
Each split's rows are hashed as canonical JSON (sorted keys, one row per line)
and compared to EXPECTED_CONTENT_SHA256; a read-back of the written parquet is
hashed the same way, so any writer that round-trips the content passes and any
content drift fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

INSTRUCTION = "Let's think step by step and output the final answer after \"####\"."
DATA_SOURCE = "openai/gsm8k"

# openai/gsm8k main split raw files under --raw-dir.
RAW_FILES = {
    "train": "main/train-00000-of-00001.parquet",
    "test": "main/test-00000-of-00001.parquet",
}

EXPECTED = {
    "train": {"rows": 7473, "content_sha256": "5cc12e9564e80f6c381dbf5687d0fd2ccd381d76dee0b6d8719b7d4942faded2"},
    "test": {"rows": 1319, "content_sha256": "233920366fcf48e8a6ea1395be8b76a07f08c34e177abc583917f36b21ffa469"},
}

SCHEMA = pa.schema(
    [
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
        (
            "extra_info",
            pa.struct(
                [
                    ("split", pa.string()),
                    ("index", pa.int64()),
                    ("answer", pa.string()),
                    ("question", pa.string()),
                ]
            ),
        ),
    ]
)


def extract_solution(solution_str: str) -> str:
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0).split("#### ")[1].replace(",", "")
    return final_solution


def content_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_rows(raw_path: Path, split: str) -> list[dict]:
    table = pq.read_table(raw_path)
    names = table.column_names
    if names != ["question", "answer"]:
        raise SystemExit(f"{raw_path}: expected raw columns [question, answer], got {names}")
    rows = []
    for idx, record in enumerate(table.to_pylist()):
        question_raw = record["question"]
        answer_raw = record["answer"]
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": question_raw + " " + INSTRUCTION}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": extract_solution(answer_raw)},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
        )
    return rows


def verify_split(rows: list[dict], split: str) -> None:
    expected = EXPECTED[split]
    if len(rows) != expected["rows"]:
        raise SystemExit(f"{split}: {len(rows)} rows, expected {expected['rows']}")
    digest = content_digest(rows)
    if digest != expected["content_sha256"]:
        raise SystemExit(f"{split}: content digest {digest} != pinned {expected['content_sha256']}")


def write_split(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), temporary)
    temporary.replace(path)


def process_split(raw_dir: Path, output_dir: Path, split: str) -> None:
    raw_path = raw_dir / RAW_FILES[split]
    if not raw_path.is_file():
        raise SystemExit(f"missing raw file: {raw_path}")
    out_path = output_dir / f"{split}.parquet"

    # A previously written parquet that still matches the pinned content is
    # reused as-is, so a rerun on the same workspace is a no-op.
    if out_path.is_file():
        try:
            existing = pq.read_table(out_path).to_pylist()
        except Exception:
            existing = None
        if existing is not None:
            verify_split(existing, split)
            print(f"gsm8k {split}: {len(existing)} rows verified in place -> {out_path}")
            return

    rows = build_rows(raw_path, split)
    verify_split(rows, split)
    write_split(rows, out_path)
    verify_split(pq.read_table(out_path).to_pylist(), split)
    print(f"gsm8k {split}: {len(rows)} rows written and verified -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, required=True, help="pinned openai/gsm8k snapshot")
    parser.add_argument("--output", type=Path, required=True, help="destination assets/gsm8k dir")
    args = parser.parse_args()
    for split in ("train", "test"):
        process_split(args.raw_dir, args.output, split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
