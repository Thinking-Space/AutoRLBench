#!/usr/bin/env python3
"""Prepare 30 canonical AIME 2024 + 30 AIME 2025 questions for mean@32.

The DAPO training parquet is linked unchanged. Validation contains each question
once: the runner must use validation n=32 and test_freq=10 (960 samples/year).
Pinned public source files are downloaded to a local cache by default.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

import prepare_dapo
from prepare_dapo import (distribution, input_manifest, load_tokenizer,
                          normalized_problem_id, reuse_existing, sha256, tokenizer_manifest)


BUNDLE = Path(__file__).resolve().parents[1]
PREPARATION_VERSION = 1
SOURCES = {
    "aime24": {
        "year": 2024, "repo_id": "HuggingFaceH4/aime_2024",
        "revision": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        "data_file": "data/train-00000-of-00001.parquet",
        "files": {
            "data/train-00000-of-00001.parquet": "26139847601a5037c237d5928b195e7260ca8074cf4f264b794af42847f79ccf",
            "README.md": "c1d82f7870ba18b505fad3027c5c6131bac989ed3e177e9fc23385b9b9b315b8",
        },
    },
    "aime25": {
        "year": 2025, "repo_id": "yentinglin/aime_2025",
        "revision": "6f71d77b0b89b9dabe07ab466c51df33f514df7f",
        "data_file": "data/train-00000-of-00001-243207c6c994e1bd.parquet",
        "files": {
            "data/train-00000-of-00001-243207c6c994e1bd.parquet": "b74b507c1a4de7c9fafd2c83dfe2ef7b80579bcf0e0a754a1765db0f21c205fc",
            "README.md": "2c72cbca643fee38cfd72c43c872285a413d49920a3dc371e96cbc101a7047bf",
        },
    },
}
ANSWER_INSTRUCTION = (
    'Please reason step by step. Write your final answer on the last line in the '
    'format "Answer: <integer>", using a plain integer without leading zeros.'
)
PROBLEM_URL = re.compile(
    r"https://artofproblemsolving\.com/wiki/index\.php/(\d{4})_AIME_(I|II)_Problems/Problem_(\d+)"
)


def fetch_source(source, filename, cache_dir, offline=False):
    path = (cache_dir / source["repo_id"].replace("/", "--") /
            source["revision"] / filename)
    expected = source["files"][filename]
    url = (f"https://huggingface.co/datasets/{source['repo_id']}/resolve/"
           f"{source['revision']}/{filename}")
    if path.exists():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Pinned source cache failed SHA256 verification: {path}")
        return path
    if offline:
        raise FileNotFoundError(f"Pinned source not cached; rerun without --offline: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AutoRLBench-AIME-preparation/1"})
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as target:
            temporary = Path(target.name)
            with urllib.request.urlopen(request, timeout=60) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    target.write(block)
        if sha256(temporary) != expected:
            raise ValueError(f"Downloaded source differs from pinned SHA256: {url}")
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)  # Atomic creation; never replace an existing cache entry.
        except FileExistsError:
            if sha256(path) != expected:
                raise ValueError(f"Concurrent source cache write failed integrity verification: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def validate_year(rows, year):
    """Reject missing/duplicate questions, wrong years and invalid integer labels."""
    if len(rows) != 30:
        raise ValueError(f"AIME {year} must contain exactly 30 canonical rows; got {len(rows)}")
    by_question = {}
    content_ids = set()
    for row in rows:
        match = PROBLEM_URL.fullmatch(str(row.get("url", "")))
        if not match or int(match[1]) != year or str(row.get("year")) != str(year):
            raise ValueError(f"AIME {year} has an inconsistent year or source URL: {row.get('url')}")
        session, number = match[2], int(match[3])
        if not 1 <= number <= 15:
            raise ValueError(f"AIME {year} has invalid problem number {number}")
        key = (session, number)
        if key in by_question:
            raise ValueError(f"Duplicate canonical AIME question: {year}/{session}/{number}")
        problem = row.get("problem")
        answer = str(row.get("answer", "")).strip()
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"AIME {year}/{session}/{number} has an empty problem")
        if not re.fullmatch(r"[0-9]{1,3}", answer) or not 0 <= int(answer) <= 999:
            raise ValueError(f"AIME {year}/{session}/{number} has an invalid integer answer")
        content_id = normalized_problem_id([{"role": "user", "content": problem}])
        if content_id in content_ids:
            raise ValueError(f"AIME {year} repeats the same normalized problem content")
        content_ids.add(content_id)
        by_question[key] = row
    expected = {(session, number) for session in ("I", "II") for number in range(1, 16)}
    if set(by_question) != expected:
        raise ValueError(f"AIME {year} is missing one or more I/II questions 1..15")
    return [(key, by_question[key]) for key in sorted(expected)]


def build_validation(source_rows, tokenizer):
    output, counts = [], {}
    all_contents = set()
    for name, source in SOURCES.items():
        rows = validate_year(source_rows[name], source["year"])
        lengths = Counter()
        for (session, number), row in rows:
            question_id = f"aime{source['year']}-{session}-{number:02d}"
            content_id = normalized_problem_id([{"role": "user", "content": row["problem"]}])
            if content_id in all_contents:
                raise ValueError("AIME 2024 and 2025 contain overlapping normalized problems")
            all_contents.add(content_id)
            prompt = [{"role": "user", "content": row["problem"] + "\n\n" + ANSWER_INSTRUCTION}]
            text = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            length = len(tokenizer.encode(text, add_special_tokens=False))
            lengths[length] += 1
            output.append({
                "prompt": prompt, "data_source": name, "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(int(str(row["answer"]).strip()))},
                "extra_info": {
                    "index": question_id, "question_id": question_id, "problem_id": question_id,
                    "source_id": str(row["id"]), "source_url": row["url"],
                    "source_repo": source["repo_id"], "source_revision": source["revision"],
                    "year": source["year"], "session": session, "problem_number": number,
                    "raw_problem": row["problem"], "problem_content_sha256": content_id,
                    "split": "val", "benchmark_profile": "dapo17k",
                    "context_bucket": "short" if length <= 512 else "long",
                    "prompt_token_length": length,
                },
            })
        counts[name] = {"year": source["year"], "rows": 30, "unique_questions": 30,
                        "sessions": {"I": 15, "II": 15}, "integer_answer_range": [0, 999],
                        "lengths_written": distribution(lengths)}
    return output, counts


def prepare(train_input, tokenizer_path, output, cache_dir, offline=False):
    train_input, tokenizer_path, output, cache_dir = [
        Path(path).expanduser().resolve() for path in (train_input, tokenizer_path, output, cache_dir)]
    if train_input.stat().st_mode & 0o222:
        raise ValueError(f"Training data must already be read-only; its source will not be changed: {train_input}")
    train_info = input_manifest(train_input)
    if train_info["rows"] < 1:
        raise ValueError("Training parquet is empty")
    tokenizer_info = tokenizer_manifest(tokenizer_path)
    provenance, source_rows = {}, {}
    for name, source in SOURCES.items():
        files = {}
        for filename in source["files"]:
            file = fetch_source(source, filename, cache_dir, offline)
            files[filename] = {
                "path": str(file), "sha256": sha256(file), "size_bytes": file.stat().st_size,
                "url": f"https://huggingface.co/datasets/{source['repo_id']}/resolve/{source['revision']}/{filename}",
            }
        data_path = files[source["data_file"]]["path"]
        source_rows[name] = pq.read_table(data_path).to_pylist()
        provenance[name] = {"repo_id": source["repo_id"], "revision": source["revision"],
                            "year": source["year"], "config": "default", "source_split": "train",
                            "usage": "validation only", "source_rows": len(source_rows[name]),
                            "files": files}
    settings = {
        "chat_template_kwargs": {"enable_thinking": False}, "add_generation_prompt": True,
        "tokenization": "apply_chat_template(tokenize=False), encode(add_special_tokens=False)",
        "prompt_suffix": ANSWER_INSTRUCTION, "label_format": "str(int(source answer)); no leading zeros",
        "train_rows": "all source rows linked unchanged", "val_rows": "30 canonical questions per year, once each",
        "length_filtering": False, "validation_row_repetition": 1,
    }
    identity = {"preparer": "prepare_aime_eval", "preparation_version": PREPARATION_VERSION,
                "script_sha256": sha256(Path(__file__)),
                "helper_script_sha256": sha256(Path(prepare_dapo.__file__)),
                "train_input": train_info, "sources": provenance,
                "tokenizer": tokenizer_info, "settings": settings,
                "libraries": {"pyarrow": pa.__version__, "transformers": prepare_dapo.transformers.__version__,
                              "tokenizers": prepare_dapo.tokenizers.__version__}}
    if output.exists():
        manifest = reuse_existing(output, identity)
        if not (output / "train.parquet").is_symlink() or (output / "train.parquet").resolve() != train_input:
            raise ValueError("Existing training output is not the expected unchanged source symlink")
        return manifest, True

    validation, coverage = build_validation(source_rows, load_tokenizer(tokenizer_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    train_link = output / "train.parquet"
    train_link.symlink_to(os.path.relpath(train_input, output))
    val_path = output / "val.parquet"
    pq.write_table(pa.Table.from_pylist(validation), val_path, compression="zstd")
    val_path.chmod(0o444)
    outputs = {
        "train": {"path": "train.parquet", "sha256": train_info["sha256"], "rows": train_info["rows"],
                  "size_bytes": train_info["size_bytes"], "link_target": os.readlink(train_link)},
        "val": {"path": "val.parquet", "sha256": sha256(val_path), "rows": len(validation),
                "size_bytes": val_path.stat().st_size},
    }
    manifest = {
        "schema_version": 3, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_profile": "dapo17k", "evaluation_profile": "aime24-aime25-mean32",
        "identity": identity, "inputs": {"train": train_info}, "sources": provenance,
        "tokenizer": tokenizer_info, "settings": settings, "outputs": outputs,
        "validation": {"output_rows": 60, "unique_questions_written": 60, "by_source": coverage,
                       "row_repetition": 1, "canonical_id_coverage_verified": True,
                       "mixed_years_in_each_source": False, "duplicate_question_ids": 0,
                       "duplicate_normalized_problem_contents": 0},
        "evaluation_protocol": {
            "samples_per_question": 32, "evaluation_interval_training_steps": 10,
            "expected_completions_per_year": 960, "expected_total_completions_per_evaluation": 1920,
            "metric": "mean@32 = arithmetic mean of 32 binary correctness scores per question, averaged over 30 questions per year",
            "runner_overrides_required": ["actor_rollout_ref.rollout.val_kwargs.n=32", "trainer.test_freq=10"],
            "execution": "This script prepares data only; the training runner must apply the protocol.",
        },
        "preservation": {"training_source_modified": False, "training_row_count_and_order_unchanged": True,
                         "validation_source_problem_text_preserved": True, "output_data_read_only": True},
        "scope": "Pinned-source data preparation only; no model generation, evaluation or GPU training performed.",
    }
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    (output / "manifest.json").chmod(0o444)
    return manifest, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=BUNDLE / "runtime/dapo17k-raw/train.parquet")
    parser.add_argument("--tokenizer", type=Path, default=BUNDLE / "assets/tokenizer")
    parser.add_argument("--output", type=Path, default=BUNDLE / "runtime/dapo17k-aime24-25")
    parser.add_argument("--cache-dir", type=Path, default=BUNDLE / "runtime/source_cache/aime")
    parser.add_argument("--offline", action="store_true", help="Require the pinned files to already exist in the cache")
    args = parser.parse_args()
    try:
        manifest, reused = prepare(args.train_input, args.tokenizer, args.output, args.cache_dir, args.offline)
    except (ValueError, FileExistsError, FileNotFoundError, urllib.error.URLError) as error:
        parser.exit(2, f"prepare_aime_eval: {error}\n")
    print(json.dumps({"status": "reused" if reused else "prepared",
                      "output": str(args.output.expanduser().resolve()),
                      "train_rows": manifest["outputs"]["train"]["rows"], "val_rows": 60,
                      "unique_questions_per_year": 30, "validation_samples_per_question": 32,
                      "evaluation_interval_training_steps": 10}, indent=2))


if __name__ == "__main__":
    main()
