#!/usr/bin/env python3
"""Assemble the task's taco_verl train/test parquets from the public sources.

Runs inside the prepare stage (networked container) after prepare.sh has
fetched BAAI/TACO and likaixin/TACO-verified, and after the author-side
purification pipeline (environment/data_prep/) has produced the committed
selection inputs in environment/selection/:

  dirty_manifest.jsonl   problems whose bundled solutions agree with each
                         other yet fail their own tests (native
                         wrong-expected-output family)
  test_manifest.jsonl    held-out-split problems whose own solutions pass
                         (verdict passes_loose under the loose budget)
  mix_spec.json          ratio, sizes, difficulty bands, seed

Layout contract (byte-compatible with the sibling task
verl_taco_difficulty_mismatch, verified against its published parquets):

  data_source  'taco'
  prompt       [{'role': 'user', 'content': PREFIX + question + STDIN_SUFFIX}]
  ability      'code'
  reward_model {'style': 'rule', 'ground_truth': json.dumps(
               {**json.loads(input_output), 'fn_name': None})}
               -- the upstream generation re-serializes the TACO field with an
               explicit fn_name key; verified positionally against 251/251 of
               the sibling task's published train rows that fall in shard 0
  extra_info   {code_type, fn_name, index, n_tests, problem_id,
                prompt_tokens, split}
  problem_id   f'train{taco_index}' / f'test{test_index}' -- the 0-based
               position in the concatenated ALL/train (resp. ALL/test) shard
               order, the sibling task's encoding

prompt_tokens counts tokenizer.apply_chat_template(..., add_generation_prompt=True)
tokens, the same quantity the launch script's data.filter_overlong_prompts
filter enforces; rows above mix_spec.max_prompt_tokens are dropped.

Only stdin problems are selected (call_based rows are excluded so that the
whole task -- purification receipts, external judge, grader -- runs one
protocol).

Determinism: the two public sources are fetched as whole pinned repos by
fetch_hf.py, selection is seeded from mix_spec.seed, and iteration order is the
stable shard order, so a rerun reproduces both parquets byte-for-byte given the
same upstream revisions and the same committed selection inputs.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taco_sources import (  # noqa: E402
    code_type_of,
    iter_taco_test,
    iter_taco_train,
    iter_verified_records,
    n_tests_of,
    sha256_file,
    verify_id_mapping,
)

# Byte-verified against the sibling task's published taco_verl parquets: the
# literal two-character \n before 'Use Standard Input format' is present in the
# upstream data generation and is reproduced here verbatim.
PREFIX = (
    "Please write a python program to address the following QUESTION. "
    "Your ANSWER should be in a code block format like this: "
    "```python # Write your code here ```. \nQUESTION:\n"
)
STDIN_SUFFIX = " \\nUse Standard Input format\nANSWER:\n"


def build_prompt(question: str) -> str:
    return PREFIX + question + STDIN_SUFFIX


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", required=True,
                    help="dir with TACO ALL/train shards and taco_verified.json")
    ap.add_argument("--manifest", required=True, help="selection/dirty_manifest.jsonl")
    ap.add_argument("--test-manifest", required=True,
                    help="selection/test_manifest.jsonl (passes_loose problems "
                         "from the held-out split)")
    ap.add_argument("--mix-spec", required=True, help="selection/mix_spec.json")
    ap.add_argument("--tokenizer", required=True, help="model dir for the chat tokenizer")
    ap.add_argument("--out", required=True, help="output dir (taco_verl)")
    args = ap.parse_args()

    sources = Path(args.sources)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mix = json.loads(Path(args.mix_spec).read_text(encoding="utf-8"))
    seed = int(mix["seed"])
    ratio = float(mix["dirty_ratio"])
    target_dirty = int(mix["target_dirty"])
    target_test = int(mix["target_test_rows"])
    train_difficulties = {d.upper() for d in mix["train_difficulties"]}
    test_difficulties = {d.upper() for d in mix["test_difficulties"]}
    max_prompt_tokens = int(mix["max_prompt_tokens"])

    manifest = {}
    with open(args.manifest, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            manifest[int(rec["taco_index"])] = rec
    if not manifest:
        raise SystemExit("error: dirty manifest is empty")

    test_manifest = {}
    with open(args.test_manifest, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            test_manifest[int(rec["test_index"])] = rec
    if not test_manifest:
        raise SystemExit("error: test manifest is empty")

    for line in verify_id_mapping(sources, sources / "taco_verified.json"):
        print("[assemble] " + line)

    verified_json = sources / "taco_verified.json"
    verified_ids = set()
    for rec in iter_verified_records(verified_json):
        verified_ids.add(int(rec["id"]))
    print(f"[assemble] verified problems: {len(verified_ids)}")

    dirty_rows, clean_rows = [], []
    for row in iter_taco_train(sources, with_solutions=False):
        if row["difficulty"] is None:
            continue
        diff = row["difficulty"].upper()
        ct = code_type_of(row["input_output"])
        nt = n_tests_of(row["input_output"])
        if ct != "stdin" or nt <= 0:
            continue
        base = {
            "taco_index": row["taco_index"],
            "name": row["name"],
            "difficulty": diff,
            "question": row["question"],
            "input_output": row["input_output"],
            "n_tests": nt,
        }
        if row["taco_index"] in manifest:
            if diff not in train_difficulties:
                continue
            rec = manifest[row["taco_index"]]
            # Cross-check the committed manifest against the fetched revision
            # by question hash -- TACO's `name` is null for most rows, so the
            # position + question bytes are the identity that matters.
            qhash = hashlib.sha256(
                row["question"].encode("utf-8", "replace")
            ).hexdigest()
            if rec.get("question_sha256") != qhash:
                raise SystemExit(
                    f"error: manifest question mismatch at taco_index "
                    f"{row['taco_index']}: committed manifest sha "
                    f"{str(rec.get('question_sha256'))[:16]} vs fetched "
                    f"{qhash[:16]} -- the committed selection does not match "
                    "the fetched TACO revision; aborting rather than mixing "
                    "wrong rows"
                )
            dirty_rows.append(base)
        # verified ids equal taco_index (0-based concatenated order),
        # verified by verify_id_mapping above
        elif row["taco_index"] in verified_ids and diff in train_difficulties:
            clean_rows.append(base)

    test_rows = []
    for row in iter_taco_test(sources, with_solutions=False):
        if row["test_index"] not in test_manifest:
            continue
        if row["difficulty"] is None or row["difficulty"].upper() not in test_difficulties:
            continue
        ct = code_type_of(row["input_output"])
        nt = n_tests_of(row["input_output"])
        if ct != "stdin" or nt <= 0:
            continue
        test_rows.append({
            "test_index": row["test_index"],
            "name": row["name"],
            "difficulty": row["difficulty"].upper(),
            "question": row["question"],
            "input_output": row["input_output"],
            "n_tests": nt,
        })

    print(f"[assemble] candidates before prompt filter: dirty={len(dirty_rows)} "
          f"clean={len(clean_rows)} test={len(test_rows)}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def prompt_tokens(rows):
        texts = [build_prompt(r["question"]) for r in rows]
        out = []
        for i in range(0, len(texts), 256):
            for text in texts[i : i + 256]:
                enc = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                )
                # transformers <5 returns the token-id list, 5.x a
                # BatchEncoding (a UserDict, so `isinstance(enc, dict)` is
                # False); accept both so the count matches the training-side
                # filter (data.max_prompt_length) either way
                ids = getattr(enc, "input_ids", None)
                if ids is None and isinstance(enc, dict):
                    ids = enc["input_ids"]
                out.append(len(ids) if ids is not None else len(enc))
        return out

    for rows in (dirty_rows, clean_rows, test_rows):
        pts = prompt_tokens(rows)
        for r, p in zip(rows, pts):
            r["prompt_tokens"] = p

    dirty_rows = [r for r in dirty_rows if r["prompt_tokens"] <= max_prompt_tokens]
    clean_rows = [r for r in clean_rows if r["prompt_tokens"] <= max_prompt_tokens]
    test_rows = [r for r in test_rows if r["prompt_tokens"] <= max_prompt_tokens]

    rng = random.Random(seed)
    if len(dirty_rows) > target_dirty:
        dirty_rows = rng.sample(dirty_rows, target_dirty)
    dirty_dist = collections.Counter(r["difficulty"] for r in dirty_rows)
    n_dirty = len(dirty_rows)
    clean_target_total = round(n_dirty * (1.0 - ratio) / ratio) if ratio else 0

    clean_selected = []
    for diff in sorted(dirty_dist):
        stratum = [r for r in clean_rows if r["difficulty"] == diff]
        want = round(clean_target_total * dirty_dist[diff] / n_dirty)
        if len(stratum) > want:
            stratum = rng.sample(stratum, want)
        clean_selected.extend(stratum)
    if len(test_rows) > target_test:
        test_rows = rng.sample(test_rows, target_test)

    train_rows = sorted(dirty_rows + clean_selected, key=lambda r: r["taco_index"])
    test_rows = sorted(test_rows, key=lambda r: r["test_index"])

    actual_dirty = len(dirty_rows)
    print(f"[assemble] train: {len(train_rows)} rows "
          f"({actual_dirty} dirty = "
          f"{actual_dirty / max(len(train_rows), 1):.1%}, target ratio {ratio:.0%}); "
          f"difficulty dist {dict(collections.Counter(r['difficulty'] for r in train_rows))}")
    print(f"[assemble] test: {len(test_rows)} rows "
          f"(difficulties {sorted(test_difficulties)}, verified-clean)")

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        ("extra_info", pa.struct([
            ("code_type", pa.string()),
            ("fn_name", pa.string()),
            ("index", pa.int64()),
            ("n_tests", pa.int64()),
            ("problem_id", pa.string()),
            ("prompt_tokens", pa.int64()),
            ("split", pa.string()),
        ])),
    ])

    def to_table(rows, split):
        # The sibling task's encoding: problem_id is the row's 0-based position
        # in the concatenated ALL/train (or ALL/test) shard order.
        def pid_of(r):
            if "taco_index" in r:
                return f"train{r['taco_index']}"
            return f"test{r['test_index']}"

        data = {
            "data_source": ["taco" for _ in rows],
            "prompt": [[{"role": "user", "content": build_prompt(r["question"])}] for r in rows],
            "ability": ["code" for _ in rows],
            "reward_model": [
                # Upstream re-serializes the TACO field with an explicit
                # fn_name key; reproduced verbatim (see module docstring).
                {
                    "ground_truth": json.dumps(
                        {**json.loads(r["input_output"]), "fn_name": None}
                    ),
                    "style": "rule",
                }
                for r in rows
            ],
            "extra_info": [
                {
                    "code_type": "stdin",
                    "fn_name": None,
                    "index": i,
                    "n_tests": r["n_tests"],
                    "problem_id": pid_of(r),
                    "prompt_tokens": r["prompt_tokens"],
                    "split": split,
                }
                for i, r in enumerate(rows)
            ],
        }
        return pa.table(data, schema=schema)

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    pq.write_table(to_table(train_rows, "train"), train_path)
    pq.write_table(to_table(test_rows, "test"), test_path)
    print(f"[assemble] wrote {train_path} ({train_path.stat().st_size} bytes, "
          f"sha256 {sha256_file(train_path)})")
    print(f"[assemble] wrote {test_path} ({test_path.stat().st_size} bytes, "
          f"sha256 {sha256_file(test_path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
