#!/usr/bin/env python3
"""Convert the local OPD JSONL contract into a VeRL RL parquet file.

The converter is intentionally independent of VeRL itself.  VeRL's
``RLHFDataset`` consumes an ordinary parquet file with the following RL
columns: ``data_source``, chat-style ``prompt``, ``ability``,
``reward_model`` and ``extra_info``.  JSONL is convenient while preparing and
auditing data, but letting Arrow infer a schema independently for every row
causes failures when (for example) one answer is a string and another is a
list.  This utility therefore uses one explicit Arrow schema and writes it in
small record batches.

The output schema keeps the fields used by VeRL in their native nested form:

* ``prompt`` is ``list<struct<role:string,content:string>>``;
* ``reward_model`` is a struct with string ``style`` and ``ground_truth``;
* ``extra_info`` is a stable struct containing the metadata used by the
  Skywork and LongRLVR verifiers.  A canonical ``raw_json`` member retains
  any additional metadata keys without making the parquet schema dynamic.
  (Known fields are stored in their native typed members and are not duplicated
  in ``raw_json``; this matters for the very large LongRLVR evidence strings.)

Ground-truth values which are not already strings are encoded as compact JSON
strings.  This is the representation expected by the released VeRL reward
managers and is lossless for JSON values.  Reward-model keys outside the
standard ``style``/``ground_truth`` pair are retained in
``reward_model.raw_json``.

Writes are transactional: a temporary parquet file and a temporary manifest
are created beside their final paths and atomically renamed only after all
validation succeeds.  Outputs are restricted to ``/historical/shared/qat`` so a
mistyped path cannot overwrite an unrelated file.  Existing files require
``--overwrite`` explicitly.

Examples::

    python scripts/convert_opd_jsonl_to_parquet.py \
        --input /historical/shared/qat/data/processed/skywork.jsonl \
        --output /historical/shared/qat/data/processed/skywork.parquet \
        --max-rows 16000

    # Verify an already-created parquet without converting it.
    python scripts/convert_opd_jsonl_to_parquet.py \
        --verify-only --output /historical/shared/qat/data/processed/skywork.parquet

Only local files are accepted.  The script does not download data and does
not read the large formal datasets unless the caller points it at them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


# Keep this constant easy to monkeypatch in tests while still resolving
# symlinks for the actual safety check.
QAT_ROOT = Path("/historical/shared/qat").resolve()


class ConversionError(ValueError):
    """A user-facing input or schema error."""


def _require_pyarrow():
    """Import pyarrow lazily so ``--help`` works in a minimal environment."""

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ConversionError(
            "pyarrow is required for JSONL to parquet conversion; use the "
            "vllm-0.19.1-test environment or install pyarrow"
        ) from exc
    return pa, pq


def _schema():
    """Return the single, versioned schema used by every output file.

    The function is used instead of module-level Arrow objects so importing
    this module remains possible without pyarrow (useful for shell help and
    lightweight unit tests).
    """

    pa, _ = _require_pyarrow()
    message = pa.struct(
        [pa.field("role", pa.string()), pa.field("content", pa.string())]
    )
    prompt = pa.list_(message)
    model_difficulty = pa.struct(
        [
            pa.field("DeepSeek-R1-Distill-Qwen-1.5B", pa.int64()),
            pa.field("DeepSeek-R1-Distill-Qwen-32B", pa.int64()),
            pa.field("DeepSeek-R1-Distill-Qwen-7B", pa.int64()),
        ]
    )
    # ``extra_info`` deliberately has a stable union of fields needed by the
    # local Skywork/LongRLVR pipelines.  ``raw_json`` is the lossless escape
    # hatch for future datasets and unknown keys.
    extra_info = pa.struct(
        [
            pa.field("index", pa.string()),
            pa.field("question", pa.string()),
            pa.field("ref_chunks", pa.list_(pa.int64())),
            pa.field("ref_chunks_text", pa.list_(pa.string())),
            pa.field("split", pa.string()),
            pa.field("model_difficulty", model_difficulty),
            pa.field("source_file", pa.string()),
            pa.field("source_row", pa.int64()),
            pa.field("source_revision", pa.string()),
            pa.field("prompt_tokens", pa.int64()),
            pa.field("original_prompt_tokens", pa.int64()),
            pa.field("truncated", pa.bool_()),
            pa.field("retained_ref_chunks", pa.list_(pa.int64())),
            pa.field("retained_chunks", pa.list_(pa.int64())),
            pa.field("reference_messages", prompt),
            pa.field("source_id", pa.string()),
            pa.field("source_length", pa.int64()),
            pa.field("verifiable", pa.bool_()),
            pa.field("difficulty", pa.int64()),
            pa.field("prompt_sha256", pa.string()),
            pa.field("raw_json", pa.string()),
        ]
    )
    reward_model = pa.struct(
        [
            pa.field("style", pa.string()),
            pa.field("ground_truth", pa.string()),
            pa.field("raw_json", pa.string()),
        ]
    )
    return pa.schema(
        [
            pa.field("example_id", pa.string()),
            pa.field("data_source", pa.string()),
            pa.field("prompt", prompt),
            pa.field("ability", pa.string()),
            pa.field("reward_model", reward_model),
            pa.field("extra_info", extra_info),
        ]
    )


def build_schema():
    """Public alias used by tests and downstream preparation scripts."""

    return _schema()


# Common aliases mirror the names used by VeRL's own data preparation
# scripts.  They are initialized lazily to avoid importing pyarrow at module
# import time; callers should use ``build_schema`` for the full output schema.
PROMPT_TYPE = None
MESSAGE_TYPE = None
REWARD_MODEL_TYPE = None
EXTRA_INFO_TYPE = None


def _json_default(value: Any) -> Any:
    """Convert a small set of scalar/container extensions to JSON values."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConversionError(f"non-finite JSON number: {value!r}")
        return value
    # This handles values accidentally supplied by a caller using numpy or
    # pyarrow while keeping the converter itself dependency-light.
    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        return _json_default(as_py())
    if isinstance(value, Mapping):
        return {str(key): _json_default(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    raise ConversionError(f"value is not JSON serializable: {type(value).__name__}")


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(
            _json_default(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConversionError(f"cannot encode value as JSON: {exc}") from exc


def _as_mapping(value: Any, *, field: str, line_number: int, allow_none: bool = True) -> dict[str, Any]:
    if value is None and allow_none:
        return {}
    if not isinstance(value, Mapping):
        raise ConversionError(
            f"line {line_number}: {field} must be an object, got {type(value).__name__}"
        )
    # JSON objects already contain JSON-compatible values.  Do not recursively
    # copy every nested evidence string here: LongRLVR rows can be hundreds of
    # kilobytes each, and the converter's low-memory guarantee depends on
    # keeping only one parsed object alive at a time.
    return {str(key): item for key, item in value.items()}


def _as_string(value: Any, *, field: str, line_number: int, allow_none: bool = True) -> str | None:
    if value is None and allow_none:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    raise ConversionError(
        f"line {line_number}: {field} must be a scalar string, got {type(value).__name__}"
    )


def _as_int(value: Any, *, field: str, line_number: int, allow_none: bool = True) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ConversionError(f"line {line_number}: {field} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConversionError(f"line {line_number}: {field} must be an integer") from exc
    # Do not silently turn 1.5 into 1.
    if isinstance(value, float) and value != integer:
        raise ConversionError(f"line {line_number}: {field} must be an integer")
    if integer < -(2**63) or integer > 2**63 - 1:
        raise ConversionError(f"line {line_number}: {field} is outside int64 range")
    return integer


def _as_bool(value: Any, *, field: str, line_number: int, allow_none: bool = True) -> bool | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        return value
    raise ConversionError(f"line {line_number}: {field} must be a boolean")


def _as_int_list(value: Any, *, field: str, line_number: int) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ConversionError(f"line {line_number}: {field} must be a list of integers")
    result: list[int] = []
    for index, item in enumerate(value):
        result.append(_as_int(item, field=f"{field}[{index}]", line_number=line_number, allow_none=False))
    return result


def _as_string_list(value: Any, *, field: str, line_number: int) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ConversionError(f"line {line_number}: {field} must be a list of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        converted = _as_string(item, field=f"{field}[{index}]", line_number=line_number, allow_none=False)
        assert converted is not None
        result.append(converted)
    return result


def _messages(value: Any, *, field: str, line_number: int) -> list[dict[str, str]]:
    """Validate and normalize a chat message list.

    A few JSONL producers write this field as a JSON-encoded string, so that
    representation is accepted as a convenience.  The emitted parquet value
    is always a real Arrow list/struct value.
    """

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConversionError(f"line {line_number}: {field} is not valid JSON") from exc
    if not isinstance(value, (list, tuple)) or not value:
        raise ConversionError(f"line {line_number}: {field} must be a non-empty list")
    result: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ConversionError(
                f"line {line_number}: {field}[{index}] must be an object"
            )
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ConversionError(
                f"line {line_number}: {field}[{index}].role must be a non-empty string"
            )
        content = message.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            # Existing preparation code uses ``str`` for harmless scalar
            # content; follow that contract instead of dropping a row.
            content = str(content)
        result.append({"role": role, "content": content})
    return result


_MODEL_DIFFICULTY_KEYS = (
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Qwen-7B",
)


def normalize_record(record: Any, *, line_number: int) -> dict[str, Any]:
    """Validate one JSON object and adapt it to :func:`build_schema`."""

    if not isinstance(record, Mapping):
        raise ConversionError(
            f"line {line_number}: top-level JSON value must be an object"
        )
    prompt = _messages(record.get("prompt"), field="prompt", line_number=line_number)
    reward_input = _as_mapping(
        record.get("reward_model"), field="reward_model", line_number=line_number
    )
    extra_input = _as_mapping(
        record.get("extra_info"), field="extra_info", line_number=line_number
    )

    ground_truth = reward_input.get("ground_truth")
    if ground_truth is None:
        ground_truth_text = ""
    elif isinstance(ground_truth, str):
        ground_truth_text = ground_truth
    else:
        ground_truth_text = _compact_json(ground_truth)

    # ``raw_json`` is an internal escape-hatch added by this converter.  When
    # validating a parquet file, it is already present in the struct and must
    # not recursively become another unknown field.
    reward_unknown = {
        key: value
        for key, value in reward_input.items()
        if key not in {"style", "ground_truth", "raw_json"}
    }

    model_input = extra_input.get("model_difficulty")
    model_values: dict[str, int | None] | None
    if model_input is None:
        model_values = None
    else:
        model_map = _as_mapping(
            model_input,
            field="extra_info.model_difficulty",
            line_number=line_number,
        )
        model_values = {
            key: _as_int(
                model_map.get(key),
                field=f"extra_info.model_difficulty.{key}",
                line_number=line_number,
            )
            for key in _MODEL_DIFFICULTY_KEYS
        }

    # Keep unknown source metadata in a deterministic JSON member.  Known
    # fields above are represented natively, so serializing them a second time
    # would unnecessarily double LongRLVR's large ref_chunks_text payload.
    _known_extra_keys = {
        "index",
        "question",
        "ref_chunks",
        "ref_chunks_text",
        "split",
        "model_difficulty",
        "source_file",
        "source_row",
        "source_revision",
        "prompt_tokens",
        "original_prompt_tokens",
        "truncated",
        "retained_ref_chunks",
        "retained_chunks",
        "reference_messages",
        "source_id",
        "source_length",
        "verifiable",
        "difficulty",
        "prompt_sha256",
    }
    unknown_extra = {
        key: value for key, value in extra_input.items() if key not in _known_extra_keys
    }
    # Preserve model-difficulty labels introduced by a future checkpoint even
    # though the fixed struct has slots for the three models used by the
    # current Skywork filter.
    if isinstance(model_input, Mapping):
        unknown_model = {
            str(key): value
            for key, value in model_input.items()
            if str(key) not in _MODEL_DIFFICULTY_KEYS
        }
        if unknown_model:
            unknown_extra["model_difficulty"] = unknown_model
    extra_typed = {
        "index": _as_string(extra_input.get("index"), field="extra_info.index", line_number=line_number),
        "question": _as_string(extra_input.get("question"), field="extra_info.question", line_number=line_number),
        "ref_chunks": _as_int_list(extra_input.get("ref_chunks"), field="extra_info.ref_chunks", line_number=line_number),
        "ref_chunks_text": _as_string_list(extra_input.get("ref_chunks_text"), field="extra_info.ref_chunks_text", line_number=line_number),
        "split": _as_string(extra_input.get("split"), field="extra_info.split", line_number=line_number),
        "model_difficulty": model_values,
        "source_file": _as_string(extra_input.get("source_file"), field="extra_info.source_file", line_number=line_number),
        "source_row": _as_int(extra_input.get("source_row"), field="extra_info.source_row", line_number=line_number),
        "source_revision": _as_string(extra_input.get("source_revision"), field="extra_info.source_revision", line_number=line_number),
        "prompt_tokens": _as_int(extra_input.get("prompt_tokens"), field="extra_info.prompt_tokens", line_number=line_number),
        "original_prompt_tokens": _as_int(extra_input.get("original_prompt_tokens"), field="extra_info.original_prompt_tokens", line_number=line_number),
        "truncated": _as_bool(extra_input.get("truncated"), field="extra_info.truncated", line_number=line_number),
        "retained_ref_chunks": _as_int_list(extra_input.get("retained_ref_chunks"), field="extra_info.retained_ref_chunks", line_number=line_number),
        "retained_chunks": _as_int_list(extra_input.get("retained_chunks"), field="extra_info.retained_chunks", line_number=line_number),
        "reference_messages": (
            _messages(
                extra_input["reference_messages"],
                field="extra_info.reference_messages",
                line_number=line_number,
            )
            if extra_input.get("reference_messages") is not None
            else None
        ),
        "source_id": _as_string(extra_input.get("source_id"), field="extra_info.source_id", line_number=line_number),
        "source_length": _as_int(extra_input.get("source_length"), field="extra_info.source_length", line_number=line_number),
        "verifiable": _as_bool(extra_input.get("verifiable"), field="extra_info.verifiable", line_number=line_number),
        "difficulty": _as_int(extra_input.get("difficulty"), field="extra_info.difficulty", line_number=line_number),
        "prompt_sha256": _as_string(extra_input.get("prompt_sha256"), field="extra_info.prompt_sha256", line_number=line_number),
        "raw_json": _compact_json(unknown_extra),
    }

    return {
        "example_id": _as_string(record.get("example_id", ""), field="example_id", line_number=line_number),
        "data_source": _as_string(record.get("data_source", "default"), field="data_source", line_number=line_number),
        "prompt": prompt,
        "ability": _as_string(record.get("ability", "math"), field="ability", line_number=line_number),
        "reward_model": {
            "style": _as_string(reward_input.get("style", "rule"), field="reward_model.style", line_number=line_number),
            "ground_truth": ground_truth_text,
            "raw_json": _compact_json(reward_unknown),
        },
        "extra_info": extra_typed,
    }


def _sha256_file(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_existing(path: str | os.PathLike[str], *, kind: str) -> Path:
    text = os.fspath(path)
    if "://" in text:
        raise ConversionError(f"{kind} must be a local path, not a URI: {text}")
    resolved = Path(text).expanduser().resolve()
    if not resolved.is_file():
        raise ConversionError(f"{kind} does not exist or is not a file: {resolved}")
    return resolved


def _under_root(path: Path, root: Path = QAT_ROOT) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_output(path: str | os.PathLike[str], *, kind: str = "output") -> Path:
    text = os.fspath(path)
    if "://" in text:
        raise ConversionError(f"{kind} must be a local path, not a URI: {text}")
    # ``resolve(strict=False)`` follows existing symlinks and normalizes the
    # path even when its final component has not been created yet.
    resolved = Path(text).expanduser().resolve(strict=False)
    if not _under_root(resolved):
        raise ConversionError(
            f"{kind} must be under {QAT_ROOT}; refusing unsafe path {resolved}"
        )
    if resolved == QAT_ROOT:
        raise ConversionError(f"{kind} must name a file below {QAT_ROOT}")
    return resolved


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _atomic_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _schema_descriptor(schema: Any) -> dict[str, Any]:
    return {
        "format": "pyarrow",
        "fields": [
            {"name": field.name, "type": str(field.type), "nullable": bool(field.nullable)}
            for field in schema
        ],
        "string": str(schema),
    }


def validate_parquet(
    path: str | os.PathLike[str],
    *,
    expected_rows: int | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Validate a VeRL parquet file without materializing it in memory.

    Returns row/column statistics useful for a manifest or a smoke test.  The
    check intentionally walks every row group and verifies the required
    nested values, not merely the parquet footer.
    """

    path = _resolve_existing(path, kind="parquet")
    if batch_size < 1:
        raise ConversionError("batch_size must be positive")
    _, pq = _require_pyarrow()
    schema = build_schema()
    parquet = pq.ParquetFile(path)
    actual = parquet.schema_arrow
    if actual != schema:
        # Arrow metadata can differ while the field layout is identical.  Use
        # a type/name comparison and give a useful error rather than accepting
        # an incompatible parquet file silently.
        actual_fields = [(field.name, str(field.type)) for field in actual]
        expected_fields = [(field.name, str(field.type)) for field in schema]
        if actual_fields != expected_fields:
            raise ConversionError(
                "parquet schema mismatch: expected "
                f"{expected_fields}, got {actual_fields}"
            )
    rows = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        for record in batch.to_pylist():
            # ``to_pylist`` turns Arrow structs into dictionaries.  Validate
            # through normalize_record as an additional contract check; it is
            # bounded by one batch and does not retain data.
            normalize_record(record, line_number=rows + 1)
            rows += 1
    if expected_rows is not None and rows != expected_rows:
        raise ConversionError(
            f"parquet row count mismatch: expected {expected_rows}, got {rows}"
        )
    return {
        "rows": rows,
        "row_groups": parquet.metadata.num_row_groups if parquet.metadata else 0,
        "schema": _schema_descriptor(schema),
    }


def convert(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    manifest_path: str | os.PathLike[str] | None = None,
    batch_size: int = 64,
    max_rows: int | None = None,
    overwrite: bool = False,
    skip_invalid: bool = False,
    compression: str = "zstd",
    verify: bool = True,
) -> dict[str, Any]:
    """Stream-convert ``input_path`` and return the written manifest.

    ``max_rows`` counts valid emitted records, so malformed rows before the
    limit are either rejected or skipped according to ``skip_invalid``.  Input
    bytes are hashed through EOF even when the output limit is reached; the
    manifest therefore identifies the complete source file.
    """

    if batch_size < 1:
        raise ConversionError("batch_size must be positive")
    if max_rows is not None and max_rows < 0:
        raise ConversionError("max_rows must be non-negative")
    input_resolved = _resolve_existing(input_path, kind="input")
    output_resolved = _resolve_output(output_path)
    if input_resolved == output_resolved:
        raise ConversionError("input and output paths must be different")
    if manifest_path is None:
        manifest_resolved = Path(str(output_resolved) + ".manifest.json")
    else:
        manifest_resolved = _resolve_output(manifest_path, kind="manifest")
    if manifest_resolved == output_resolved:
        raise ConversionError("manifest and output paths must be different")
    if manifest_resolved == input_resolved:
        raise ConversionError("manifest must not overwrite the input JSONL")
    if output_resolved.exists():
        try:
            if os.path.samefile(input_resolved, output_resolved):
                raise ConversionError("output must not be the same file as the input")
        except FileNotFoundError:
            # The existence check above can race with another creator; the
            # eventual atomic replace remains safe for a newly-created path.
            pass

    if not overwrite:
        existing = [path for path in (output_resolved, manifest_resolved) if path.exists()]
        if existing:
            names = ", ".join(str(path) for path in existing)
            raise ConversionError(f"output exists; pass --overwrite to replace it: {names}")

    pa, pq = _require_pyarrow()
    schema = build_schema()
    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    manifest_resolved.parent.mkdir(parents=True, exist_ok=True)

    # Hash input while reading it, avoiding a separate full-file pass.  The
    # output hash is computed after the atomic rename.
    input_digest = hashlib.sha256()
    input_bytes = 0
    input_lines = 0
    parsed_rows = 0
    emitted_rows = 0
    skipped_rows = 0
    validation_errors: list[dict[str, Any]] = []
    validation_errors_truncated = 0
    batch: list[dict[str, Any]] = []
    writer = None
    temporary_output: Path | None = None

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_resolved.name}.", suffix=".tmp", dir=str(output_resolved.parent)
    )
    os.close(fd)
    temporary_output = Path(temporary_name)
    try:
        writer = pq.ParquetWriter(
            str(temporary_output), schema=schema, compression=compression
        )
        with input_resolved.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                input_digest.update(raw_line)
                input_bytes += len(raw_line)
                if not raw_line.strip():
                    continue
                input_lines += 1

                # Once max_rows is reached, continue reading only to complete
                # the source hash.  This keeps the first-N contract while
                # making the manifest's input SHA identify the whole file.
                if max_rows is not None and emitted_rows >= max_rows:
                    continue
                try:
                    decoded = json.loads(raw_line)
                    parsed_rows += 1
                    normalized = normalize_record(decoded, line_number=line_number)
                except (json.JSONDecodeError, UnicodeDecodeError, ConversionError) as exc:
                    if not skip_invalid:
                        if isinstance(exc, ConversionError):
                            raise
                        raise ConversionError(f"line {line_number}: invalid JSON: {exc}") from exc
                    skipped_rows += 1
                    # Keep malformed-row diagnostics bounded as well.  A bad
                    # file with millions of lines must not defeat the
                    # streaming memory guarantee just because
                    # ``--skip-invalid`` was requested.
                    if len(validation_errors) < 100:
                        validation_errors.append({"line": line_number, "error": str(exc)})
                    else:
                        validation_errors_truncated += 1
                    continue
                batch.append(normalized)
                emitted_rows += 1
                if len(batch) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()
        writer.close()
        writer = None
        _fsync_file(temporary_output)
        os.replace(temporary_output, output_resolved)
        temporary_output = None
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if temporary_output is not None:
            try:
                temporary_output.unlink()
            except FileNotFoundError:
                pass
        raise

    verification = validate_parquet(
        output_resolved, expected_rows=emitted_rows, batch_size=batch_size
    ) if verify else {"rows": emitted_rows, "schema": _schema_descriptor(schema)}
    output_size = output_resolved.stat().st_size
    output_sha = _sha256_file(output_resolved)
    input_sha = input_digest.hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "converter": "convert_opd_jsonl_to_parquet",
        "input": {
            "path": str(input_resolved),
            "sha256": input_sha,
            "size_bytes": input_bytes,
            "nonempty_lines": input_lines,
            "parsed_rows_before_limit": parsed_rows,
        },
        "output": {
            "path": str(output_resolved),
            "sha256": output_sha,
            "size_bytes": output_size,
            "rows": emitted_rows,
        },
        # Flat aliases make the manifest easy to consume from shell tools and
        # preserve the names used by older preparation manifests.
        "input_sha256": input_sha,
        "output_sha256": output_sha,
        "input_path": str(input_resolved),
        "output_path": str(output_resolved),
        "input_rows": input_lines,
        "output_rows": emitted_rows,
        "rows": emitted_rows,
        "schema": _schema_descriptor(schema),
        "schema_string": str(schema),
        "schema_fields": [field.name for field in schema],
        "parameters": {
            "batch_size": batch_size,
            "max_rows": max_rows,
            "overwrite": overwrite,
            "skip_invalid": skip_invalid,
            "compression": compression,
            "verify": verify,
        },
        "validation": {
            "skipped_rows": skipped_rows,
            "errors": validation_errors,
            "errors_truncated": validation_errors_truncated,
            "result": verification,
        },
    }
    _atomic_manifest(manifest_resolved, manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream OPD JSONL into a fixed-schema VeRL parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "--input-jsonl",
        dest="input_path",
        help="local OPD JSONL input",
    )
    parser.add_argument(
        "--output",
        "--output-parquet",
        dest="output_path",
        help="output parquet (must be under /historical/shared/qat)",
    )
    parser.add_argument(
        "--manifest",
        help="manifest path (defaults to <output>.manifest.json; also under the QAT root)",
    )
    # LongRLVR prompts are large (often >100 KiB of text).  A modest default
    # keeps peak Python/Arrow memory bounded while callers can raise it for
    # short math-only files.
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="emit the deterministic first N valid records; still hash input through EOF",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="skip malformed rows and record them in the manifest (default: fail fast)",
    )
    parser.add_argument("--compression", default="zstd")
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="skip the post-write parquet validation pass",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate an existing parquet; do not read JSONL or write anything",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_rows is not None and args.max_rows < 0:
        parser.error("--max-rows must be non-negative")
    if args.verify_only:
        if not args.output_path:
            parser.error("--verify-only requires --output")
    elif not args.input_path or not args.output_path:
        parser.error("--input and --output are required unless --verify-only is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.verify_only:
            result = validate_parquet(args.output_path, batch_size=args.batch_size)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            result = convert(
                args.input_path,
                args.output_path,
                manifest_path=args.manifest,
                batch_size=args.batch_size,
                max_rows=args.max_rows,
                overwrite=args.overwrite,
                skip_invalid=args.skip_invalid,
                compression=args.compression,
                verify=args.verify,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except (ConversionError, OSError) as exc:
        # Keep CLI errors concise and make shell failures unambiguous.
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
