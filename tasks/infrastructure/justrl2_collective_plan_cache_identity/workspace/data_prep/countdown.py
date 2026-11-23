"""Convert Jiayi-Pan/Countdown-Tasks-3to4 into JustRL-II / Miles JSONL splits."""
from __future__ import annotations

import argparse
import json
import os
import zlib
from pathlib import Path

TRAIN_ROWS = 8192
TEST_ROWS = 1024

PROMPT_TEMPLATE = (
    "Solve the Countdown arithmetic puzzle. Using the given numbers, form an arithmetic "
    "expression with +, -, *, /, and parentheses that equals the target integer. "
    "Use each given number exactly once. Keep your reasoning brief and output only the "
    "final expression (without '= target') inside <answer>...</answer> tags.\n\n"
    "Example 1:\n"
    "Numbers: [12, 3, 5]\n"
    "Target: 21\n"
    "Reasoning: 12 - 5 = 7, and 3 * 7 = 21.\n"
    "<answer>3 * (12 - 5)</answer>\n\n"
    "Example 2:\n"
    "Numbers: [8, 4, 2, 6]\n"
    "Target: 26\n"
    "Reasoning: 8 * 2 = 16, and 16 + 6 + 4 = 26.\n"
    "<answer>8 * 2 + 6 + 4</answer>\n\n"
    "Example 3:\n"
    "Numbers: [25, 7, 4, 2]\n"
    "Target: 43\n"
    "Reasoning: 7 * 2 = 14, 25 + 14 = 39, and 39 + 4 = 43.\n"
    "<answer>25 + 7 * 2 + 4</answer>\n\n"
    "Now solve:\n"
    "Numbers: {nums}\n"
    "Target: {target}\n"
    "Reasoning:"
)


def make_row(target: int, numbers: list[int], split: str, idx: int) -> dict:
    content = PROMPT_TEMPLATE.format(nums=numbers, target=target)
    return {
        "prompt": [{"role": "user", "content": content}],
        "label": json.dumps({"numbers": numbers, "target": target}, sort_keys=True),
        "metadata": {
            "split": split,
            "index": idx,
            "data_source": "Jiayi-Pan/Countdown-Tasks-3to4",
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def from_parquet(src: Path, out: Path) -> None:
    import pyarrow.parquet as pq

    parquet_path = src if src.is_file() else src / "data" / "train-00000-of-00001.parquet"
    table = pq.read_table(parquet_path)
    total = table.num_rows
    assert total >= TRAIN_ROWS + TEST_ROWS, f"Need >= {TRAIN_ROWS + TEST_ROWS} rows, got {total}"
    targets = table["target"].to_pylist()
    nums_list = table["nums"].to_pylist()

    train_rows = [
        make_row(int(targets[i]), [int(x) for x in nums_list[i]], "train", i)
        for i in range(TRAIN_ROWS)
    ]
    test_start = total - TEST_ROWS
    test_rows = [
        make_row(int(targets[i]), [int(x) for x in nums_list[i]], "test", i)
        for i in range(test_start, total)
    ]
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "test.jsonl", test_rows)
    print(f"train: {len(train_rows)} rows, test: {len(test_rows)} rows (disjoint from {total} total)")


def pack_binary(src_parquet: Path, out_bin: Path) -> None:
    """Pack exact TRAIN_ROWS + TEST_ROWS integer tuples for air-gapped cluster staging."""
    import pyarrow.parquet as pq

    table = pq.read_table(src_parquet)
    total = table.num_rows
    targets = table["target"].to_pylist()
    nums_list = table["nums"].to_pylist()
    buf = bytearray()
    buf.extend(total.to_bytes(4, "little"))
    buf.extend(TRAIN_ROWS.to_bytes(4, "little"))
    buf.extend(TEST_ROWS.to_bytes(4, "little"))
    for idx in list(range(TRAIN_ROWS)) + list(range(total - TEST_ROWS, total)):
        ns = [int(x) for x in nums_list[idx]]
        buf.append(int(targets[idx]))
        buf.append(len(ns))
        for x in ns:
            buf.append(x)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_bin.write_bytes(zlib.compress(bytes(buf), 9))
    print(f"Packed {TRAIN_ROWS}+{TEST_ROWS} rows from {total} -> {out_bin} ({out_bin.stat().st_size} bytes)")


def unpack_binary(src_bin: Path, out: Path) -> None:
    raw = zlib.decompress(src_bin.read_bytes())
    total = int.from_bytes(raw[0:4], "little")
    n_train = int.from_bytes(raw[4:8], "little")
    n_test = int.from_bytes(raw[8:12], "little")
    pos = 12
    train_rows = []
    for i in range(n_train):
        target = raw[pos]
        k = raw[pos + 1]
        nums = list(raw[pos + 2 : pos + 2 + k])
        pos += 2 + k
        train_rows.append(make_row(target, nums, "train", i))
    test_start = total - n_test
    test_rows = []
    for offset in range(n_test):
        target = raw[pos]
        k = raw[pos + 1]
        nums = list(raw[pos + 2 : pos + 2 + k])
        pos += 2 + k
        test_rows.append(make_row(target, nums, "test", test_start + offset))
    assert pos == len(raw), f"trailing bytes: {pos} != {len(raw)}"
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "test.jsonl", test_rows)
    print(f"Unpacked train: {len(train_rows)} rows, test: {len(test_rows)} rows (total={total})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, help="Countdown snapshot dir or parquet file")
    ap.add_argument("--src-bin", type=Path, help="Packed binary countdown file")
    ap.add_argument("--pack-to", type=Path, help="Pack parquet to compact binary")
    ap.add_argument("--out", type=Path, help="Output directory for train.jsonl and test.jsonl")
    args = ap.parse_args()
    if args.pack_to:
        assert args.src, "--src required with --pack-to"
        pack_binary(args.src, args.pack_to)
    elif args.src_bin:
        assert args.out, "--out required with --src-bin"
        unpack_binary(args.src_bin, args.out)
    else:
        assert args.src and args.out, "--src and --out required"
        from_parquet(args.src, args.out)


if __name__ == "__main__":
    main()
