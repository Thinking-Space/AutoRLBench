#!/usr/bin/env python3
"""Prepare verifiable prompts for the KV-OPD/VeRL experiments.

The script deliberately writes *JSONL* rather than a framework-specific cache.
VeRL can ingest the records after a small dataset wrapper, while the JSONL is
also easy to inspect and to hash.  Raw downloads stay under ``/historical/shared``;
the default output root is ``/historical/shared/qat/data/processed``.

Examples (the commands are intentionally explicit about filtering):

  python scripts/prepare_opd_data.py skywork --ability math \
      --output /historical/shared/qat/data/processed/skywork_math.jsonl

  python scripts/prepare_opd_data.py longalign --tokenizer \
      /historical/shared/qat/models/base/qwen2.5-7b-instruct-1m \
      --max-prompt-tokens 12288

  python scripts/prepare_opd_data.py longrlvr --max-prompt-tokens 12288 \
      --tokenizer /historical/shared/qat/models/base/qwen2.5-7b-instruct-1m

No command downloads data.  A parquet dependency (``pyarrow``) is required
only for the Skywork and LongRLVR modes; a Transformers tokenizer is optional
and is required only when an exact token-length filter is requested.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


QAT_ROOT = Path("/historical/shared/qat").resolve()
DEFAULTS = {
    "skywork": QAT_ROOT / "data/raw/skywork-or1-rl-data/data/*.parquet",
    "longalign": QAT_ROOT / "data/raw/longalign-10k/long.jsonl",
    "longrlvr": QAT_ROOT / "data/raw/longrlvr-data/qwen2.5-7b/train_data_64k.parquet",
}
DATASET_INFO = {
    "skywork": {
        "repo": "Skywork/Skywork-OR1-RL-Data",
        "revision": "1cdedc52e0e2db85fdf252f9be682e63a5a38c33",
    },
    "longalign": {
        "repo": "zai-org/LongAlign-10k",
        "revision": "12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc",
    },
    "longrlvr": {
        "repo": "Guanzheng/LongRLVR-Data",
        "revision": "282447325bc6034eeb39107598fe025ecc2474cd",
    },
}


class UsageError(RuntimeError):
    """An input/schema error that should be shown without a traceback."""


def _jsonable(value: Any) -> Any:
    """Convert Arrow scalar/container values to ordinary JSON values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    # pyarrow scalar objects expose ``as_py``.  Keep this optional so that the
    # JSONL/LongAlign path does not import pyarrow.
    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        return _jsonable(as_py())
    # Do not fail on a harmless scalar extension type.  This is preferable to
    # silently dropping a field, and makes the manifest disclose the value.
    return str(value)


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _expand_paths(items: Sequence[str], dataset: str) -> List[Path]:
    paths: List[Path] = []
    for item in items:
        matches = sorted(glob.glob(item))
        if not matches and Path(item).exists():
            matches = [item]
        if not matches:
            raise UsageError("input path/pattern does not exist: %s" % item)
        for match in matches:
            path = Path(match).expanduser().resolve()
            if not path.is_file():
                raise UsageError("input is not a file: %s" % path)
            paths.append(path)
    # A shell-expanded glob can overlap with another argument.  Preserve order
    # but avoid processing the same physical file twice.
    unique: List[Path] = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if dataset == "longalign" and len(unique) != 1:
        raise UsageError("LongAlign mode expects one JSONL file")
    return unique


def _revision_from_sidecar(path: Path, fallback: str) -> str:
    """Read the HF cache revision sidecar when available.

    ``huggingface_hub`` stores the commit SHA as the first line of a nearby
    ``*.metadata`` file.  This is only provenance; a missing sidecar is valid.
    """

    for parent in [path.parent] + list(path.parents):
        cache = parent / ".cache/huggingface/download"
        if not cache.is_dir():
            continue
        candidates = list(cache.rglob(path.name + ".metadata"))
        if not candidates:
            # Parquet files are often in a subdirectory beneath download/.
            candidates = list(cache.rglob(path.stem + ".metadata"))
        for sidecar in candidates:
            try:
                first = sidecar.read_text(encoding="utf-8").splitlines()[0].strip()
            except (OSError, IndexError):
                continue
            if first:
                return first
    return fallback


def _stable_id(dataset: str, path: Path, row_number: int, payload: Any) -> str:
    material = {
        "dataset": dataset,
        "file": str(path),
        "row": row_number,
        "payload": payload,
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "%s-%s" % (dataset, hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24])


def _parse_json_string(value: Any) -> Tuple[Any, bool]:
    if not isinstance(value, str):
        return value, False
    try:
        return json.loads(value), True
    except (TypeError, ValueError, json.JSONDecodeError):
        return value, False


class TokenCounter:
    """Small optional adapter around a fast Transformers tokenizer."""

    def __init__(self, model_path: Optional[str]) -> None:
        self.path = model_path
        self.tokenizer = None
        if model_path:
            try:
                from transformers import AutoTokenizer  # type: ignore

                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            except Exception as exc:  # pragma: no cover - depends on env/network
                raise UsageError(
                    "could not load tokenizer %r; install transformers or omit "
                    "--max-prompt-tokens: %s" % (model_path, exc)
                ) from exc

    @property
    def enabled(self) -> bool:
        return self.tokenizer is not None

    def count(self, messages: Sequence[Dict[str, Any]]) -> int:
        if self.tokenizer is None:
            raise UsageError("a tokenizer is required for exact token filtering")
        # Counting a prompt without a generation marker is intentional: the
        # response budget is controlled by VeRL at rollout time.  Chat-template
        # support is preferred, but a plain role/content serialization keeps the
        # utility usable with older tokenizers.
        tok = self.tokenizer
        try:
            ids = tok.apply_chat_template(
                list(messages), tokenize=True, add_generation_prompt=True
            )
            if isinstance(ids, dict):
                ids = ids["input_ids"]
            return len(ids)
        except Exception:
            text = "\n".join(
                "%s: %s" % (m.get("role", ""), m.get("content", ""))
                for m in messages
            )
            return len(tok(text, add_special_tokens=True)["input_ids"])

    def encode_text(self, text: str) -> List[int]:
        """Encode raw content without chat-template wrappers."""

        if self.tokenizer is None:
            raise UsageError("a tokenizer is required for token-aware truncation")
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode_text(self, ids: Sequence[int]) -> str:
        if self.tokenizer is None:
            raise UsageError("a tokenizer is required for token-aware truncation")
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def truncate_text(self, text: str, max_tokens: int, mode: str = "head_tail") -> str:
        """Truncate raw text while retaining both ends where possible."""

        if max_tokens < 1:
            return ""
        ids = self.encode_text(text)
        if len(ids) <= max_tokens:
            return text
        if mode == "head":
            kept = ids[:max_tokens]
        else:
            # Reserve a small, tokenized marker.  The exact final prompt length
            # is checked by the caller, so an occasional one-token adjustment
            # is harmless.
            head = max_tokens // 2
            tail = max_tokens - head
            kept = ids[:head] + ids[-tail:]
        return self.decode_text(kept)


_CHUNK_RE = re.compile(
    r"<CHUNK_(?P<id>\d+)>.*?</CHUNK_(?P=id)>", re.DOTALL
)


def _truncate_longrlvr_content(
    content: str,
    reference_chunk_ids: Sequence[Any],
    counter: TokenCounter,
    max_prompt_tokens: int,
    mode: str,
) -> Tuple[str, List[int], bool]:
    """Keep the question and referenced chunks in a long RLVR prompt.

    LongRLVR puts the question after a sequence of ``<CHUNK_i>`` blocks and
    supplies the evidence indices in ``extra_info.ref_chunks``.  A plain
    head/tail cut can remove all evidence.  In ``evidence`` mode we first keep
    referenced blocks (in source order), then fill remaining budget with
    alternating head/tail blocks; the final suffix (normally the question) is
    always retained.  If the source has no parseable chunk tags, this falls
    back to a token-level head/tail cut.
    """

    original_ids = counter.encode_text(content)
    # Account for the model's chat wrapper and generation marker.  Measuring
    # an empty user prompt is stable for the local Qwen/DeepSeek tokenizers.
    wrapper = counter.count([{"role": "user", "content": ""}])
    content_budget = max(1, max_prompt_tokens - wrapper)
    if len(original_ids) <= content_budget:
        return content, [], False
    if mode == "head_tail":
        cut = counter.truncate_text(content, content_budget)
        return cut, [], True

    matches = list(_CHUNK_RE.finditer(content))
    if not matches:
        cut = counter.truncate_text(content, content_budget)
        return cut, [], True

    prefix = content[: matches[0].start()]
    suffix = content[matches[-1].end() :]
    chunks = [(int(m.group("id")), m.group(0)) for m in matches]
    by_id = {chunk_id: text for chunk_id, text in chunks}
    refs: List[int] = []
    for value in reference_chunk_ids or []:
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value in by_id and value not in refs:
            refs.append(value)

    # Keep the question suffix first.  It is normally short, but if malformed
    # input makes it enormous, retaining its tail is preferable to dropping
    # the actual question.
    suffix_ids = counter.encode_text(suffix)
    if len(suffix_ids) >= content_budget:
        cut = counter.truncate_text(suffix, content_budget)
        return cut, [], True
    remaining = content_budget - len(suffix_ids)

    # The instruction header is useful but less important than evidence.  Cap
    # it so that a pathological preamble cannot consume the complete budget.
    prefix_ids = counter.encode_text(prefix)
    prefix_keep = min(len(prefix_ids), min(512, remaining))
    prefix_text = counter.decode_text(prefix_ids[:prefix_keep])
    remaining -= prefix_keep

    selected: Dict[int, str] = {}

    def add_chunk(chunk_id: int) -> None:
        nonlocal remaining
        if remaining <= 0 or chunk_id in selected:
            return
        raw = by_id[chunk_id]
        ids = counter.encode_text(raw)
        if len(ids) <= remaining:
            selected[chunk_id] = raw
            remaining -= len(ids)
            return
        # A referenced block can itself be larger than the residual budget.
        # Keep its beginning and end, but never consume the reserved suffix.
        if remaining >= 8:
            selected[chunk_id] = counter.truncate_text(raw, remaining)
            remaining = 0

    for chunk_id in refs:
        add_chunk(chunk_id)

    # Fill spare capacity with head/tail chunks, alternating sides.  This is
    # deterministic and keeps broad document coverage without random sampling.
    order: List[int] = []
    left, right = 0, len(chunks) - 1
    take_left = True
    while left <= right:
        candidate = chunks[left][0] if take_left else chunks[right][0]
        if candidate not in order:
            order.append(candidate)
        if take_left:
            left += 1
        else:
            right -= 1
        take_left = not take_left
    for chunk_id in order:
        add_chunk(chunk_id)
        if remaining <= 0:
            break

    selected_ids = set(selected)
    selected_parts = [text for chunk_id, text in chunks if chunk_id in selected_ids]
    assembled = prefix_text
    if selected_parts:
        assembled += "\n" + "\n".join(selected_parts) + "\n"
    assembled += suffix
    # Account for tokenization artifacts introduced by decoding/concatenation.
    # A final head/tail fallback is only a last resort; the suffix remains in
    # the tail and all retained evidence is recorded in metadata.
    if counter.count([{"role": "user", "content": assembled}]) > max_prompt_tokens:
        assembled = counter.truncate_text(assembled, content_budget)
    return assembled, sorted(selected_ids), True


def _messages(value: Any) -> List[Dict[str, Any]]:
    value, _ = _parse_json_string(value)
    if not isinstance(value, list):
        raise UsageError("messages/prompt is not a list")
    result: List[Dict[str, Any]] = []
    for message in value:
        if not isinstance(message, dict):
            raise UsageError("a message is not an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise UsageError("message role is missing or not a string")
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        result.append({"role": role, "content": content})
    if not result:
        raise UsageError("empty prompt/messages")
    return result


def _write_jsonl(
    records: Iterable[Dict[str, Any]],
    output: Path,
    overwrite: bool,
) -> Tuple[int, Dict[str, int]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise UsageError("output exists; pass --overwrite to replace it: %s" % output)
    fd, temporary = tempfile.mkstemp(
        prefix=".%s." % output.name, suffix=".tmp", dir=str(output.parent)
    )
    counts = {"written": 0, "serialization_errors": 0}
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                try:
                    handle.write(
                        json.dumps(
                            _jsonable(record),
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    counts["written"] += 1
                except (TypeError, ValueError, OverflowError):
                    counts["serialization_errors"] += 1
                    raise
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return counts["written"], counts


def _source_manifest(paths: Sequence[Path], revisions: Dict[Path, str]) -> List[Dict[str, Any]]:
    result = []
    for path in paths:
        result.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "revision": revisions.get(path),
            }
        )
    return result


def _prepare_skywork(
    paths: Sequence[Path],
    args: argparse.Namespace,
    tokenizer: TokenCounter,
    stats: Dict[str, Any],
) -> Iterator[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise UsageError("Skywork mode requires pyarrow") from exc

    columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    for path in paths:
        try:
            parquet = pq.ParquetFile(path)
        except Exception as exc:
            raise UsageError("cannot open parquet %s: %s" % (path, exc)) from exc
        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
            for row in batch.to_pylist():
                if args.max_rows is not None and stats["written"] >= args.max_rows:
                    return
                row_number = stats["seen"]
                stats["seen"] += 1
                ability = row.get("ability")
                if args.ability and ability not in args.ability:
                    stats["skipped_ability"] += 1
                    continue
                info = _jsonable(row.get("extra_info") or {})
                difficulty = (info.get("model_difficulty") or {}).get(args.difficulty_model)
                if difficulty is None or not (
                    args.difficulty_min <= int(difficulty) <= args.difficulty_max
                ):
                    stats["skipped_difficulty"] += 1
                    continue
                try:
                    prompt = _messages(row.get("prompt"))
                except UsageError:
                    stats["skipped_schema"] += 1
                    continue
                prompt_tokens = None
                if tokenizer.enabled:
                    prompt_tokens = tokenizer.count(prompt)
                    if args.max_prompt_tokens is not None and prompt_tokens > args.max_prompt_tokens:
                        stats["skipped_length"] += 1
                        continue
                elif args.max_prompt_tokens is not None:
                    # Skywork has no declared prompt length.  Refuse an
                    # apparently exact filter rather than silently using chars.
                    raise UsageError(
                        "Skywork --max-prompt-tokens requires --tokenizer "
                        "(there is no source length field)"
                    )
                reward = _jsonable(row.get("reward_model") or {})
                ground_truth, parsed = _parse_json_string(reward.get("ground_truth"))
                if parsed:
                    reward["ground_truth"] = ground_truth
                    stats["ground_truth_json"] += 1
                else:
                    stats["ground_truth_plain"] += 1
                # This is the cleanup in Skywork's official preprocessing
                # script; code test cases with null values are not useful to a
                # verifier and are removed, while math lists/dicts are intact.
                if ability == "code" and isinstance(reward.get("ground_truth"), dict):
                    reward["ground_truth"] = {
                        k: v for k, v in reward["ground_truth"].items() if v is not None
                    }
                source_index = info.get("index")
                record = {
                    "example_id": _stable_id("skywork", path, row_number, prompt),
                    "data_source": row.get("data_source"),
                    "prompt": prompt,
                    "ability": ability,
                    "reward_model": reward,
                    "extra_info": {
                        **info,
                        "source_file": str(path),
                        "source_row": row_number,
                        "source_revision": _revision_from_sidecar(
                            path, DATASET_INFO["skywork"]["revision"]
                        ),
                    },
                }
                if prompt_tokens is not None:
                    record["extra_info"]["prompt_tokens"] = prompt_tokens
                stats["written"] += 1
                yield record


def _prepare_longalign(
    path: Path,
    args: argparse.Namespace,
    tokenizer: TokenCounter,
    stats: Dict[str, Any],
) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            if not line.strip():
                continue
            if args.max_rows is not None and stats["written"] >= args.max_rows:
                return
            stats["seen"] += 1
            try:
                row = json.loads(line)
                messages = _messages(row.get("messages"))
            except (ValueError, TypeError, UsageError):
                stats["skipped_schema"] += 1
                continue
            # OPD is online: only the prefix before the first assistant answer
            # is presented to the policy.  Keep the answer in metadata for an
            # optional SFT/evaluation side path; it is not a reward target.
            first_assistant = next(
                (i for i, message in enumerate(messages) if message["role"] == "assistant"),
                len(messages),
            )
            prompt = messages[:first_assistant]
            if not prompt:
                stats["skipped_schema"] += 1
                continue
            prompt_tokens = tokenizer.count(prompt) if tokenizer.enabled else None
            if args.max_prompt_tokens is not None:
                if prompt_tokens is not None:
                    if prompt_tokens > args.max_prompt_tokens:
                        if args.truncate == "none":
                            stats["skipped_length"] += 1
                            continue
                        # LongAlign has no evidence-index field.  The generic
                        # head/tail mode is therefore the only meaningful
                        # choice; ``evidence`` intentionally falls back to it.
                        content = prompt[0].get("content", "")
                        wrapper = tokenizer.count([{"role": "user", "content": ""}])
                        content = tokenizer.truncate_text(
                            content,
                            max(1, args.max_prompt_tokens - wrapper),
                        )
                        prompt = [dict(prompt[0], content=content)] + prompt[1:]
                        prompt_tokens = tokenizer.count(prompt)
                        if prompt_tokens > args.max_prompt_tokens:
                            stats["skipped_length"] += 1
                            continue
                        stats["truncated_rows"] += 1
                elif args.use_declared_length:
                    declared = row.get("length")
                    if isinstance(declared, (int, float)) and declared > args.max_prompt_tokens:
                        stats["skipped_length"] += 1
                        continue
                else:
                    raise UsageError(
                        "LongAlign --max-prompt-tokens requires --tokenizer, or "
                        "explicit --use-declared-length (the source length is approximate)"
                    )
            reference = messages[first_assistant:]
            extra = {
                "source_id": row.get("id"),
                "source_length": row.get("length"),
                "reference_messages": reference,
                "source_file": str(path),
                "source_row": row_number,
                "source_revision": _revision_from_sidecar(
                    path, DATASET_INFO["longalign"]["revision"]
                ),
                "verifiable": False,
            }
            if prompt_tokens is not None:
                extra["prompt_tokens"] = prompt_tokens
            record = {
                "example_id": _stable_id("longalign", path, row_number, row.get("id")),
                "data_source": "LongAlign-10k",
                "prompt": prompt,
                "ability": "long_context",
                "extra_info": extra,
            }
            stats["written"] += 1
            yield record


def _prepare_longrlvr(
    path: Path,
    args: argparse.Namespace,
    tokenizer: TokenCounter,
    stats: Dict[str, Any],
) -> Iterator[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise UsageError("LongRLVR mode requires pyarrow") from exc
    columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
        for row in batch.to_pylist():
            if args.max_rows is not None and stats["written"] >= args.max_rows:
                return
            row_number = stats["seen"]
            stats["seen"] += 1
            try:
                prompt = _messages(row.get("prompt"))
            except UsageError:
                stats["skipped_schema"] += 1
                continue
            info = _jsonable(row.get("extra_info") or {})
            prompt_tokens = tokenizer.count(prompt) if tokenizer.enabled else None
            original_prompt_tokens = prompt_tokens
            retained_chunks: List[int] = []
            truncated = False
            if args.max_prompt_tokens is not None:
                if prompt_tokens is None:
                    raise UsageError(
                        "LongRLVR --max-prompt-tokens requires --tokenizer"
                    )
                if prompt_tokens > args.max_prompt_tokens:
                    if args.truncate == "none":
                        stats["skipped_length"] += 1
                        continue
                    content = prompt[0].get("content", "")
                    content, retained_chunks, truncated = _truncate_longrlvr_content(
                        content,
                        info.get("ref_chunks") or [],
                        tokenizer,
                        args.max_prompt_tokens,
                        args.truncate,
                    )
                    prompt = [dict(prompt[0], content=content)] + prompt[1:]
                    prompt_tokens = tokenizer.count(prompt)
                    # A malformed tokenizer/template can still exceed the
                    # budget after truncation.  Never emit an over-budget row.
                    if prompt_tokens > args.max_prompt_tokens:
                        stats["skipped_length"] += 1
                        continue
                    stats["truncated_rows"] += 1
            stats["prompt_tokens_total"] += int(prompt_tokens or 0)
            stats["original_prompt_tokens_total"] += int(original_prompt_tokens or 0)
            extra = {
                **info,
                "source_file": str(path),
                "source_row": row_number,
                "source_revision": _revision_from_sidecar(
                    path, DATASET_INFO["longrlvr"]["revision"]
                ),
            }
            if prompt_tokens is not None:
                extra["prompt_tokens"] = prompt_tokens
            if original_prompt_tokens is not None:
                extra["original_prompt_tokens"] = original_prompt_tokens
            extra["truncated"] = bool(truncated)
            if retained_chunks:
                extra["retained_ref_chunks"] = retained_chunks
            record = {
                "example_id": _stable_id("longrlvr", path, row_number, prompt),
                "data_source": row.get("data_source"),
                "prompt": prompt,
                "ability": row.get("ability"),
                "reward_model": _jsonable(row.get("reward_model") or {}),
                "extra_info": extra,
            }
            stats["written"] += 1
            yield record


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DEFAULTS))
    parser.add_argument(
        "--input", nargs="+", dest="inputs", help="local file(s) or shell glob(s)"
    )
    parser.add_argument("--output", help="processed JSONL path (under /historical/shared/qat by default)")
    parser.add_argument("--manifest", help="manifest path; defaults to <output>.manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--tokenizer", help="local HF tokenizer for exact prompt token counts")
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument(
        "--truncate",
        choices=("none", "head_tail", "evidence"),
        default="none",
        help=(
            "when a prompt exceeds --max-prompt-tokens: skip it (none), keep "
            "both token ends (head_tail), or for LongRLVR retain ref_chunks "
            "and the question before filling head/tail (evidence)"
        ),
    )
    parser.add_argument("--use-declared-length", action="store_true", help="LongAlign only; use source length as an approximate filter")
    parser.add_argument("--allow-output-outside-root", action="store_true")
    parser.add_argument("--no-source-hash", action="store_true", help="skip expensive input SHA256 hashing")
    parser.add_argument("--ability", action="append", choices=["math", "code"], help="Skywork ability filter; repeat for both")
    parser.add_argument("--difficulty-model", default="DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--difficulty-min", type=int, default=1)
    parser.add_argument("--difficulty-max", type=int, default=15)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    if args.max_prompt_tokens is not None and args.max_prompt_tokens < 1:
        parser.error("--max-prompt-tokens must be positive")
    if args.difficulty_min > args.difficulty_max:
        parser.error("difficulty range is inverted")
    if args.dataset != "skywork" and args.ability:
        parser.error("--ability is only valid in skywork mode")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    raw_inputs = args.inputs or [str(DEFAULTS[args.dataset])]
    paths = _expand_paths(raw_inputs, args.dataset)
    default_output = QAT_ROOT / ("data/processed/%s_opd.jsonl" % args.dataset)
    output = Path(args.output or default_output).expanduser().resolve()
    if not args.allow_output_outside_root:
        try:
            output.relative_to(QAT_ROOT)
        except ValueError as exc:
            raise UsageError(
                "refusing output outside %s; pass --allow-output-outside-root explicitly" % QAT_ROOT
            ) from exc
    if output in paths:
        raise UsageError("output must not overwrite an input file")
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else Path(str(output) + ".manifest.json")
    if not args.allow_output_outside_root:
        try:
            manifest.relative_to(QAT_ROOT)
        except ValueError as exc:
            raise UsageError("manifest is outside the guarded QAT root") from exc
    # Check both destinations before consuming any input.  This avoids the
    # surprising partial-update case where a pre-existing manifest is noticed
    # only after the JSONL has already been replaced.
    if output.exists() and not args.overwrite:
        raise UsageError("output exists; pass --overwrite to replace it: %s" % output)
    if manifest.exists() and not args.overwrite:
        raise UsageError("manifest exists; pass --overwrite to replace it: %s" % manifest)

    tokenizer = TokenCounter(args.tokenizer)
    stats: Dict[str, Any] = {
        "seen": 0,
        "written": 0,
        "skipped_ability": 0,
        "skipped_difficulty": 0,
        "skipped_length": 0,
        "skipped_schema": 0,
        "ground_truth_json": 0,
        "ground_truth_plain": 0,
        "truncated_rows": 0,
        "prompt_tokens_total": 0,
        "original_prompt_tokens_total": 0,
    }
    if args.dataset == "skywork":
        records = _prepare_skywork(paths, args, tokenizer, stats)
    elif args.dataset == "longalign":
        records = _prepare_longalign(paths[0], args, tokenizer, stats)
    else:
        if len(paths) != 1:
            raise UsageError("LongRLVR mode expects one parquet file")
        records = _prepare_longrlvr(paths[0], args, tokenizer, stats)
    written, write_stats = _write_jsonl(records, output, args.overwrite)
    stats.update(write_stats)

    revisions = {path: _revision_from_sidecar(path, DATASET_INFO[args.dataset]["revision"]) for path in paths}
    source_files = [] if args.no_source_hash else _source_manifest(paths, revisions)
    manifest_data = {
        "schema_version": 1,
        "generated_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "dataset_mode": args.dataset,
        "source_dataset": DATASET_INFO[args.dataset]["repo"],
        "source_revision": DATASET_INFO[args.dataset]["revision"],
        "source_files": source_files,
        "source_hashes_skipped": bool(args.no_source_hash),
        "output": {
            "path": str(output),
            "rows": written,
            "size_bytes": output.stat().st_size,
            "sha256": _sha256(output),
        },
        "filters": {
            "ability": args.ability,
            "difficulty_model": args.difficulty_model if args.dataset == "skywork" else None,
            "difficulty_min": args.difficulty_min if args.dataset == "skywork" else None,
            "difficulty_max": args.difficulty_max if args.dataset == "skywork" else None,
            "max_prompt_tokens": args.max_prompt_tokens,
            "truncate": args.truncate,
            "tokenizer": str(Path(args.tokenizer).resolve()) if args.tokenizer else None,
            "declared_length_used": bool(args.use_declared_length),
        },
        "counts": stats,
        "record_contract": {
            "required": ["example_id", "data_source", "prompt", "ability"],
            "optional": ["reward_model", "extra_info"],
            "prompt": "list of {role: string, content: string}; OPD receives prompt only",
            "longalign_note": "reference_messages are metadata only; no verifier/reward is provided",
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and not args.overwrite:
        raise UsageError("manifest exists; pass --overwrite to replace it: %s" % manifest)
    fd, temporary = tempfile.mkstemp(prefix=".%s." % manifest.name, suffix=".tmp", dir=str(manifest.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest_data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, manifest)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    print(json.dumps({"output": str(output), "manifest": str(manifest), "counts": stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UsageError as exc:
        raise SystemExit("error: %s" % exc)
