#!/usr/bin/env python3
"""Copy and annotate every existing DAPO/AIME parquet row in source order.

Duplicate rows, conflicting labels, split overlaps and long prompts are retained.
Only prompt statistics and cached lengths are kept in memory between batches.
Source files and prompt/answer text are never edited.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq
import tokenizers
import transformers
from transformers import AutoTokenizer


PROFILE = "dapo17k"
PREPARATION_VERSION = 2
MAX_PROMPT_TOKENS = 2048
SHORT_PROMPT_TOKENS = 512
BATCH_SIZE = 1024
SOURCE_ID_EXAMPLES = 5
EMPTY_THINK_SUFFIX = re.compile(r"<\|im_start\|>assistant\s*<think>\s*</think>\s*\Z")
ANNOTATIONS = {
    "context_bucket": pa.string(),
    "prompt_token_length": pa.int64(),
    "problem_id": pa.string(),
    "benchmark_profile": pa.string(),
    "source_id": pa.string(),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def input_manifest(path):
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    parquet = pq.ParquetFile(path)
    digest = sha256(path)
    if (path.stat().st_size, path.stat().st_mtime_ns) != before:
        raise RuntimeError(f"Source changed while computing its fingerprint: {path}")
    return {"path": str(path), "sha256": digest,
            "size_bytes": before[0], "rows": parquet.metadata.num_rows}


def tokenizer_manifest(path):
    # Include all local files that can affect tokenizer loading or its template,
    # while deliberately excluding the (possibly very large) model weights.
    names = {"config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json", "vocab.json",
             "vocab.txt", "merges.txt", "tokenizer.model", "spiece.model",
             "sentencepiece.bpe.model", "chat_template.jinja"}
    files = {}
    for file in sorted(path.rglob("*")):
        if file.is_file() and (file.name in names or file.suffix == ".jinja"):
            files[str(file.relative_to(path))] = {
                "sha256": sha256(file), "size_bytes": file.stat().st_size}
    if "tokenizer_config.json" not in files:
        raise ValueError(f"Missing tokenizer_config.json in local tokenizer directory: {path}")
    return {"path": str(path), "sha256": json_digest(files), "files": files}


def normalized_problem_id(prompt):
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a nonempty list of role/content messages")
    messages = []
    for message in prompt:
        if not isinstance(message, dict):
            raise ValueError("prompt messages must be objects")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Only text messages with string role/content are supported")
        # NFC avoids converting mathematical compatibility characters (e.g.
        # superscripts) into different expressions. Original text is retained.
        messages.append({"role": unicodedata.normalize("NFC", role).strip(),
                         "content": " ".join(unicodedata.normalize("NFC", content).split())})
    return json_digest(messages)


def canonical_label(row):
    reward = row.get("reward_model")
    label = reward.get("ground_truth") if isinstance(reward, dict) else None
    if not isinstance(label, str) or not re.fullmatch(r"[+-]?[0-9]+", label.strip()):
        raise ValueError("reward_model.ground_truth must be a string containing an integer")
    # Compare integer values without changing the original string in the data.
    return str(int(label.strip()))


def source_id(row, input_path, row_index):
    extra = row.get("extra_info")
    if extra is not None and not isinstance(extra, dict):
        raise ValueError("extra_info must be a struct or null")
    index = (extra or {}).get("index")
    return str(index) if index is not None else f"{input_path.name}:row:{row_index}"


def copy_split(path, split, destination, tokenizer, groups, token_cache):
    parquet = pq.ParquetFile(path)
    required = {"prompt", "reward_model"}
    if not required.issubset(parquet.schema_arrow.names):
        raise ValueError(f"{split} is missing required columns: {required - set(parquet.schema_arrow.names)}")
    problem_ids = set()
    length_counts = Counter()
    row_index = 0
    schema = output_schema(parquet.schema_arrow)
    with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE):
            rows = batch.to_pylist()
            for row in rows:
                try:
                    problem_id = normalized_problem_id(row["prompt"])
                    label = canonical_label(row)
                    identifier = source_id(row, path, row_index)
                except (TypeError, ValueError, KeyError) as error:
                    raise ValueError(f"Invalid {split} source row {row_index}: {error}") from error
                problem_ids.add(problem_id)
                group = groups.setdefault(problem_id, {"labels": set(), "splits": {}})
                group["labels"].add(label)
                if split not in group["splits"]:
                    group["splits"][split] = {
                        "count": 0, "labels": Counter(),
                        "source_id_examples": [], "raw_label_examples": []}
                occurrence = group["splits"][split]
                occurrence["count"] += 1
                occurrence["labels"][label] += 1
                for key, value in (("source_id_examples", identifier),
                                   ("raw_label_examples", row["reward_model"]["ground_truth"])):
                    if value not in occurrence[key] and len(occurrence[key]) < SOURCE_ID_EXAMPLES:
                        occurrence[key].append(value)
                # Whitespace variants share a reporting ID but can have different
                # token lengths. Cache only completely identical message lists.
                cache_key = json_digest(row["prompt"])
                if cache_key not in token_cache:
                    rendered = tokenizer.apply_chat_template(
                        row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    token_cache[cache_key] = len(tokenizer.encode(rendered, add_special_tokens=False))
                length = token_cache[cache_key]
                length_counts[length] += 1
                row["extra_info"] = dict(row.get("extra_info") or {})
                row["extra_info"].update(
                    context_bucket="short" if length <= SHORT_PROMPT_TOKENS else "long",
                    prompt_token_length=length, problem_id=problem_id,
                    benchmark_profile=PROFILE, source_id=identifier)
                row_index += 1
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    if row_index != parquet.metadata.num_rows:
        raise ValueError(f"{split} metadata row count differs from rows actually read")
    destination.chmod(0o444)
    return problem_ids, length_counts


def output_schema(source):
    fields = []
    if "extra_info" in source.names:
        extra_type = source.field("extra_info").type
        if pa.types.is_struct(extra_type):
            fields = [field for field in extra_type if field.name not in ANNOTATIONS]
        elif not pa.types.is_null(extra_type):
            raise ValueError("extra_info must have an Arrow struct type")
    fields.extend(pa.field(name, kind) for name, kind in ANNOTATIONS.items())
    extra_field = pa.field("extra_info", pa.struct(fields))
    if "extra_info" in source.names:
        result = source.set(source.get_field_index("extra_info"), extra_field)
    else:
        result = source.append(extra_field)
    # Pandas schema metadata, if present, describes the old extra_info struct.
    metadata = dict(result.metadata or {})
    metadata.pop(b"pandas", None)
    return result.with_metadata(metadata or None)


def distribution(length_counts):
    if not length_counts:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "p50": None, "p90": None, "p95": None, "p99": None,
                "at_most_512": 0, "over_512": 0, "over_2048": 0}
    ordered = sorted(length_counts)
    count = sum(length_counts.values())
    result = {"count": count, "min": ordered[0], "max": ordered[-1],
              "mean": sum(value * length_counts[value] for value in ordered) / count,
              "at_most_512": sum(length_counts[value] for value in ordered if value <= SHORT_PROMPT_TOKENS),
              "over_512": sum(length_counts[value] for value in ordered if value > SHORT_PROMPT_TOKENS),
              "over_2048": sum(length_counts[value] for value in ordered if value > MAX_PROMPT_TOKENS)}
    for percentile in (50, 90, 95, 99):
        rank = math.ceil(count * percentile / 100)
        cumulative = 0
        for value in ordered:
            cumulative += length_counts[value]
            if cumulative >= rank:
                result[f"p{percentile}"] = value
                break
    return result


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    control = [{"role": "user", "content": "What is 2 + 2?"}]
    disabled = tokenizer.apply_chat_template(control, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    enabled = tokenizer.apply_chat_template(control, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=True)
    if not EMPTY_THINK_SUFFIX.search(disabled) or EMPTY_THINK_SUFFIX.search(enabled):
        raise ValueError("Tokenizer must support Qwen3 enable_thinking=False with a closed empty thinking prefix")
    return tokenizer


def reuse_existing(output, identity):
    manifest_path = output / "manifest.json"
    if not output.is_dir() or not manifest_path.is_file():
        raise FileExistsError(f"Refusing existing output without a complete manifest: {output}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("identity") != identity:
        raise FileExistsError(f"Existing output has different input/tokenizer/preparation identity: {output}")
    for split in ("train", "val"):
        recorded = manifest.get("outputs", {}).get(split, {})
        file = output / f"{split}.parquet"
        if (recorded.get("path") != file.name or not file.is_file()
                or file.stat().st_size != recorded.get("size_bytes")
                or sha256(file) != recorded.get("sha256")
                or pq.ParquetFile(file).metadata.num_rows != recorded.get("rows")):
            raise FileExistsError(f"Existing output failed integrity verification: {file}")
        if file.stat().st_mode & 0o222:
            raise FileExistsError(f"Existing prepared data is no longer read-only: {file}")
    return manifest


def prepare(train_input, val_input, tokenizer_path, output):
    train_input, val_input, tokenizer_path, output = [
        Path(path).expanduser().resolve() for path in (train_input, val_input, tokenizer_path, output)]
    if train_input == val_input:
        raise ValueError("Training and validation inputs must be different files")
    if not tokenizer_path.is_dir():
        raise ValueError(f"--tokenizer must be a local tokenizer directory: {tokenizer_path}")
    inputs = {split: input_manifest(path) for split, path in (("train", train_input), ("val", val_input))}
    initial_stats = {path: (path.stat().st_size, path.stat().st_mtime_ns)
                     for path in (train_input, val_input)}
    tokenizer_info = tokenizer_manifest(tokenizer_path)
    settings = {
        "benchmark_profile": PROFILE, "max_prompt_tokens": None,
        "length_reporting_threshold": MAX_PROMPT_TOKENS,
        "short_prompt_max_tokens": SHORT_PROMPT_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False}, "add_generation_prompt": True,
        "tokenization": "apply_chat_template(tokenize=False), encode(add_special_tokens=False)",
        "identity_normalization": "ordered role/content messages; Unicode NFC; content whitespace collapsed; role stripped",
        "label_conflicts": "different integer values for a normalized prompt, across either split",
        "deduplication": False, "label_conflict_removal": False,
        "train_val_exclusion": False, "length_filtering": False,
        "row_order": "preserve every source row in original order",
        "batch_size": BATCH_SIZE,
    }
    identity = {"preparation_version": PREPARATION_VERSION, "script_sha256": sha256(Path(__file__)),
                "inputs": inputs, "tokenizer": tokenizer_info, "settings": settings,
                "libraries": {"pyarrow": pa.__version__, "transformers": transformers.__version__,
                              "tokenizers": tokenizers.__version__}}
    if output.exists():
        manifest = reuse_existing(output, identity)
        return manifest, True

    tokenizer = load_tokenizer(tokenizer_path)
    # Exclusive creation prevents replacing either complete or partial output.
    # A failed run leaves a directory without a manifest; reuse then refuses it.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    groups, token_cache, split_ids, lengths, outputs = {}, {}, {}, {}, {}
    for split, path in (("train", train_input), ("val", val_input)):
        print(f"Copying {split}: all {inputs[split]['rows']} source rows in batches of {BATCH_SIZE}", file=sys.stderr)
        file = output / f"{split}.parquet"
        split_ids[split], lengths[split] = copy_split(path, split, file, tokenizer, groups, token_cache)
        actual_rows = pq.ParquetFile(file).metadata.num_rows
        if actual_rows != inputs[split]["rows"]:
            raise RuntimeError(f"{split}: copying changed the source row count")
        outputs[split] = {"path": file.name, "sha256": sha256(file),
                          "size_bytes": file.stat().st_size, "rows": actual_rows}
    for path, state in initial_stats.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != state:
            raise RuntimeError(f"Source changed during preparation: {path}")

    conflicts = []
    conflict_ids = {key for key, group in groups.items() if len(group["labels"]) > 1}
    for problem_id in sorted(conflict_ids):
        group = groups[problem_id]
        conflicts.append({
            "problem_id": problem_id, "canonical_integer_labels": sorted(group["labels"]),
            "splits": {split: {"source_rows": value["count"],
                               "label_row_counts": dict(value["labels"]),
                               "raw_label_examples": value["raw_label_examples"],
                               "source_id_examples": value["source_id_examples"]}
                       for split, value in group["splits"].items()}})

    overlap_ids = split_ids["train"] & split_ids["val"]
    statistics = {}
    for split in ("train", "val"):
        split_conflicts = split_ids[split] & conflict_ids
        split_overlap = split_ids[split] & overlap_ids
        length_summary = distribution(lengths[split])
        statistics[split] = {
            "source_rows": inputs[split]["rows"], "unique_prompt_count": len(split_ids[split]),
            "duplicate_source_rows": inputs[split]["rows"] - len(split_ids[split]),
            "duplicate_rows_removed": 0,
            "conflicting_unique_prompts_observed": len(split_conflicts),
            "conflicting_source_rows_observed": sum(groups[key]["splits"][split]["count"] for key in split_conflicts),
            "conflicting_source_rows_removed": 0,
            "train_val_overlap_unique_prompts_observed": len(split_overlap),
            "train_val_overlap_source_rows_observed": sum(groups[key]["splits"][split]["count"] for key in split_overlap),
            "train_val_overlap_source_rows_removed": 0,
            "over_2048_source_rows_observed": length_summary["over_2048"],
            "over_2048_source_rows_removed": 0,
            "output_rows": outputs[split]["rows"], "lengths_written": length_summary,
            "output_prompts_over_512": length_summary["over_512"],
        }
    manifest = {
        "schema_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_profile": PROFILE, "identity": identity, "inputs": inputs,
        "tokenizer": tokenizer_info, "settings": settings, "statistics": statistics,
        "tokenization_cache": {"exact_unique_prompts_tokenized": len(token_cache),
                               "total_rows_annotated": sum(item["rows"] for item in outputs.values()),
                               "key": "SHA256 of complete, unnormalized prompt messages"},
        "validation": {
            "source_rows": inputs["val"]["rows"], "output_rows": outputs["val"]["rows"],
            "unique_questions_before_conflict": len(split_ids["val"]),
            "unique_questions_written": len(split_ids["val"]),
            "source_repeated_rows": inputs["val"]["rows"] - len(split_ids["val"]),
            "note": "All AIME source rows, including repeated copies, are retained. Row count is not the number of independent questions.",
        },
        "label_conflicts": {"action": "retain_all_rows", "conflicting_unique_prompts": len(conflicts),
                            "conflicts": conflicts},
        "train_val_overlap": {"action": "retain_all_rows", "shared_unique_prompts": len(overlap_ids),
                              "shared_problem_ids": sorted(overlap_ids), "excluded_rows": 0},
        "length_filter": {"enabled": False, "reporting_threshold_tokens": MAX_PROMPT_TOKENS,
                          "rows_removed": 0},
        "outputs": outputs,
        "preservation": {"original_prompt_messages": True, "original_reward_model": True,
                         "original_row_order": True, "original_row_count": True,
                         "duplicate_rows": True, "conflicting_labels": True,
                         "train_val_overlaps": True, "overlong_prompts": True,
                         "source_files_modified": False, "output_data_read_only": True},
        "scope": "Data preparation only; no model generation or GPU training was performed.",
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    manifest_path.chmod(0o444)
    return manifest, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--val-input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Local Qwen3 tokenizer directory; no downloads are performed")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory, or an identical previously prepared directory")
    args = parser.parse_args()
    try:
        manifest, reused = prepare(args.train_input, args.val_input, args.tokenizer, args.output)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"prepare_dapo: {error}\n")
    print(json.dumps({"status": "reused" if reused else "prepared",
                      "output": str(args.output.expanduser().resolve()),
                      "train_rows": manifest["outputs"]["train"]["rows"],
                      "val_rows": manifest["outputs"]["val"]["rows"],
                      "val_unique_questions": manifest["validation"]["unique_questions_written"],
                      "manifest": str(args.output.expanduser().resolve() / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
