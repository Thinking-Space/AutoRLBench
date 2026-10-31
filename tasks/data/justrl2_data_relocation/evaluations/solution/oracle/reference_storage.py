"""Stdlib-only storage for an explicitly labeled, isolated data-loading fault.

Difficulty describes historical source accuracy, never measured training results.
"""

import errno
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _content_hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalized(value):
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    return value


def _validate_prompt_label(row, context):
    if not isinstance(row, dict) or "label" not in row:
        raise ValueError(f"{context}: expected an object containing prompt and label")
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt or any(
        not isinstance(message, dict) or not isinstance(message.get("role"), str) or "content" not in message
        for message in prompt
    ):
        raise ValueError(f"{context}: prompt must be a nonempty list of chat messages")


def _source_candidates(source):
    groups = {"easy": [], "hard": []}
    seen = set()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"Invalid source JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Source line {line_number} is not a JSON object")
            metadata = row.get("metadata")
            accuracy = metadata.get("accuracy") if isinstance(metadata, dict) else None
            if type(accuracy) not in (int, float) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                continue
            difficulty = "easy" if accuracy >= 0.875 else "hard" if accuracy <= 0.125 else None
            if difficulty is None:
                continue
            _validate_prompt_label(row, f"Source line {line_number}")
            prompt_key = _content_hash(_normalized(row["prompt"]))
            if prompt_key in seen:
                continue
            seen.add(prompt_key)
            identifier = f"{difficulty}-{line_number:08d}"
            payload = {
                "id": identifier, "prompt": row["prompt"], "label": row["label"],
                "metadata": {
                    "difficulty": difficulty, "source_line": line_number,
                    "historical_accuracy": accuracy,
                },
            }
            provenance = {
                "id": identifier, "difficulty": difficulty, "source_line": line_number,
                "source_row_id": row.get("id", line_number),
                "source_line_sha256": hashlib.sha256(raw_line).hexdigest(),
                "prompt_sha256": _content_hash(row["prompt"]),
                "label_sha256": _content_hash(row["label"]),
                "content_sha256": _content_hash({"prompt": row["prompt"], "label": row["label"]}),
            }
            compact = len(json.dumps(row["prompt"], ensure_ascii=False, separators=(",", ":"))) < 1600
            groups[difficulty].append((payload, provenance, compact))
    return groups, digest.hexdigest()


def _require_empty_output(output):
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise FileExistsError(f"Output must be absent or empty: {output}")


def _write_new(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def prepare(source: Path, output: Path, train_per_group=32, eval_per_group=8, seed=20261031) -> dict:
    """Copy disjoint historical easy/hard rows; return the final CASE.json marker.

    Prompt identity normalizes whitespace in string values without changing the
    copied prompt. The first eligible source occurrence wins globally, including
    when duplicate prompts have conflicting historical difficulty labels.
    """
    if type(train_per_group) is not int or train_per_group < 1:
        raise ValueError("train_per_group must be a positive integer")
    if type(eval_per_group) is not int or eval_per_group < 0:
        raise ValueError("eval_per_group must be a nonnegative integer")
    source = Path(source).resolve(strict=True)
    output = Path(output)
    _require_empty_output(output)
    groups, source_hash = _source_candidates(source)
    rng = random.Random(seed)
    total = train_per_group + eval_per_group
    for difficulty, candidates in groups.items():
        if len(candidates) < total:
            raise ValueError(f"Insufficient unique {difficulty} examples: need {total}, found {len(candidates)}")
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: not item[2])
        groups[difficulty] = candidates[:total]

    # Check again before creating anything; never merge with a populated store.
    _require_empty_output(output)
    output.mkdir(parents=True, exist_ok=True)
    for difficulty in groups:
        (output / "live" / difficulty).mkdir(parents=True)
    manifest_rows = []
    provenance_rows = []
    evaluation = []
    for index in range(total):
        split = "train" if index < train_per_group else "eval"
        for difficulty in ("easy", "hard"):
            payload, provenance, _ = groups[difficulty][index]
            provenance_rows.append({**provenance, "split": split})
            if split == "eval":
                evaluation.append(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                continue
            relative = f"live/{difficulty}/{payload['id']}.json"
            encoded = _json_bytes(payload)
            _write_new(output / relative, encoded)
            manifest_rows.append({
                "id": payload["id"], "difficulty": difficulty, "path": relative,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "source_line": provenance["source_line"], "split": "train",
            })
    _write_new(output / "manifest.json", _json_bytes({"schema_version": 1, "rows": manifest_rows}))
    _write_new(output / "eval.jsonl", "".join(evaluation).encode("utf-8"))
    if sha256_file(source) != source_hash:
        raise ValueError("Source changed during preparation; no successful CASE marker was written")
    marker = {
        "schema_version": 1, "synthetic_fault_injection": True,
        "source": {"path": str(source), "sha256": source_hash},
        "seed": seed, "train_per_group": train_per_group, "eval_per_group": eval_per_group,
        "difficulty_basis": {
            "field": "metadata.accuracy", "easy_min": 0.875, "hard_max": 0.125,
            "description": "Historical source accuracy; not current model performance.",
            "prompt_deduplication": "Whitespace-normalized string values; first eligible source row wins.",
            "compact_prompt_preference": "Serialized prompt length < 1600 Unicode characters.",
        },
        "rows": provenance_rows,
    }
    _write_new(output / "CASE.json", _json_bytes(marker))
    return marker


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _decode_json(raw, context):
    try:
        return json.loads(raw, parse_constant=_reject_constant)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON in {context}") from exc


def _require_schema(value, context):
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError(f"Invalid schema_version in {context}")


def _validate_record(record):
    if not isinstance(record, dict):
        raise ValueError("Manifest row must be an object")
    identifier = record.get("id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError("Invalid record ID")
    if record.get("difficulty") not in ("easy", "hard") or record.get("split") != "train":
        raise ValueError("Manifest rows must be easy/hard training records")
    if type(record.get("source_line")) is not int or record["source_line"] < 1:
        raise ValueError("Invalid source_line in manifest")
    if not isinstance(record.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
        raise ValueError("Invalid SHA256 in manifest")
    if not isinstance(record.get("path"), str) or not record["path"]:
        raise ValueError("Record path must be a nonempty relative path")


class BackingStore:
    """Resolve backing files on every read; missing data is never cached or hidden."""

    def __init__(self, root: Path):
        root = Path(root)
        if root.is_symlink():
            raise ValueError("Store root must not be a symlink")
        self.root = root.resolve(strict=True)
        self._check_marker()
        self._manifest, _ = self._load_manifest()
        self.rows = self._manifest["rows"]

    def _path(self, relative):
        if not isinstance(relative, str) or not relative or "\x00" in relative:
            raise ValueError("Expected a nonempty relative path")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Path must stay under the store root")
        if self.root.is_symlink():
            raise ValueError("Store root became a symlink")
        current = self.root
        for component in path.parts:
            current /= component
            if current.is_symlink():
                raise ValueError(f"Symlink traversal is forbidden: {relative}")
        try:
            current.resolve().relative_to(self.root)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"Path escapes the store root: {relative}") from exc
        return current

    def _read_bytes(self, relative):
        self._path(relative)
        descriptors = []
        try:
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            current = os.open(self.root, directory_flags)
            descriptors.append(current)
            parts = Path(relative).parts
            if not parts:
                raise ValueError("Expected a regular file")
            for component in parts[:-1]:
                current = os.open(component, directory_flags, dir_fd=current)
                descriptors.append(current)
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
            descriptors.append(descriptor)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"Expected a regular file: {relative}")
            with os.fdopen(descriptors.pop(), "rb") as stream:
                return stream.read()
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise ValueError(f"Unsafe symlink or non-directory path: {relative}") from exc
            if exc.errno == errno.ENOENT:
                raise FileNotFoundError(errno.ENOENT, exc.strerror, str(self.root / relative)) from exc
            raise
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def _check_marker(self):
        marker = _decode_json(self._read_bytes("CASE.json"), "CASE.json")
        _require_schema(marker, "CASE.json")
        if marker.get("synthetic_fault_injection") is not True:
            raise ValueError("CASE.json must explicitly mark synthetic_fault_injection=true")

    def _load_manifest(self):
        raw = self._read_bytes("manifest.json")
        manifest = _decode_json(raw, "manifest.json")
        _require_schema(manifest, "manifest.json")
        if not isinstance(manifest.get("rows"), list) or not manifest["rows"]:
            raise ValueError("Manifest must contain a nonempty rows array")
        identifiers = set()
        paths = set()
        for record in manifest["rows"]:
            _validate_record(record)
            self._path(record["path"])
            if record["id"] in identifiers or record["path"] in paths:
                raise ValueError("Duplicate record ID or path in manifest")
            identifiers.add(record["id"])
            paths.add(record["path"])
        return manifest, raw

    def read(self, record) -> dict:
        _validate_record(record)
        raw = self._read_bytes(record["path"])
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"SHA256 hash mismatch for {record['id']}")
        payload = _decode_json(raw, record["path"])
        _validate_prompt_label(payload, record["path"])
        if payload.get("id") != record["id"]:
            raise ValueError(f"Payload ID does not match record ID: {record['id']}")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or any(
            metadata.get(field) != record[field] for field in ("difficulty", "source_line")
        ):
            raise ValueError(f"Payload metadata does not match manifest: {record['id']}")
        accuracy = metadata.get("historical_accuracy")
        if type(accuracy) not in (int, float) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"Invalid historical_accuracy for {record['id']}")
        if (record["difficulty"] == "easy" and accuracy < 0.875) or (
            record["difficulty"] == "hard" and accuracy > 0.125
        ):
            raise ValueError(f"Historical accuracy does not match difficulty: {record['id']}")
        return payload

    def _atomic_write(self, relative, raw):
        destination = self._path(relative)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            self._path(relative)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _migration_files(self, manifest):
        files = []
        hard_parents = set()
        for row in manifest["rows"]:
            expected = f"live/{row['difficulty']}/{row['id']}.json"
            if row["difficulty"] == "easy":
                if row["path"] != expected:
                    raise ValueError("Unexpected easy backing path")
                continue
            destination = f"archive/hard/{row['id']}.json"
            if row["path"] not in (expected, destination):
                raise ValueError("Unexpected hard backing path")
            hard_parents.add(str(Path(row["path"]).parent))
            files.append({"id": row["id"], "sha256": row["sha256"],
                          "source": expected, "destination": destination})
        if not files or len(hard_parents) != 1:
            raise ValueError("Expected a nonempty hard group with one consistent mapping")
        return files, hard_parents == {"archive/hard"}

    def _validate_hard_directory(self, manifest, relative):
        directory = self._path(relative)
        if not directory.is_dir():
            if not directory.exists():
                raise FileNotFoundError(2, "Missing hard backing directory", str(directory))
            raise ValueError("Hard backing path is not a directory")
        hard = [row for row in manifest["rows"] if row["difficulty"] == "hard"]
        for row in hard:
            self.read({**row, "path": f"{relative}/{row['id']}.json"})
        if {path.name for path in directory.iterdir()} != {f"{row['id']}.json" for row in hard}:
            raise ValueError("Hard directory contains unowned files; refusing to move them")

    def _receipt(self, manifest, before, after, action, repair):
        files, _ = self._migration_files(manifest)
        return {
            "action": action, "repair": repair, "rows_moved": len(files), "files": files,
            "source": "live/hard", "destination": "archive/hard",
            "manifest_before_sha256": hashlib.sha256(before).hexdigest(),
            "manifest_after_sha256": hashlib.sha256(after).hexdigest(),
        }

    def _load_journal(self, files):
        path = self._path("relocation.json")
        if not path.exists():
            return {"schema_version": 1, "events": [], "pending": None}
        journal = _decode_json(self._read_bytes("relocation.json"), "relocation.json")
        _require_schema(journal, "relocation.json")
        events = journal.get("events")
        if not isinstance(events, list) or len(events) > 2 or "pending" not in journal:
            raise ValueError("Invalid relocation journal")
        for index, event in enumerate(events):
            if not isinstance(event, dict) or event.get("files") != files or event.get("rows_moved") != len(files):
                raise ValueError("Relocation event does not match backing files")
            if event.get("source") != "live/hard" or event.get("destination") != "archive/hard":
                raise ValueError("Invalid relocation event paths")
            if event.get("action") != ("relocate" if index == 0 else "repair") or type(event.get("repair")) is not bool:
                raise ValueError("Invalid relocation event action")
            for key in ("manifest_before_sha256", "manifest_after_sha256"):
                if not isinstance(event.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", event[key]):
                    raise ValueError("Invalid manifest hash in relocation event")
            if index and (not event["repair"] or events[index - 1]["repair"] or
                          event["manifest_before_sha256"] != events[index - 1]["manifest_after_sha256"]):
                raise ValueError("Invalid relocation event sequence")
        return journal

    def _complete_pending(self, journal, current_raw):
        pending = journal["pending"]
        if not isinstance(pending, dict) or not all(
            isinstance(pending.get(key), str) for key in ("manifest_before", "manifest_after")
        ) or not isinstance(pending.get("receipt"), dict):
            raise ValueError("Invalid pending relocation")
        before = pending["manifest_before"].encode("utf-8")
        after = pending["manifest_after"].encode("utf-8")
        manifest = _decode_json(before, "pending manifest")
        _require_schema(manifest, "pending manifest")
        if not isinstance(manifest.get("rows"), list):
            raise ValueError("Invalid pending manifest rows")
        for row in manifest["rows"]:
            _validate_record(row)
        receipt = pending["receipt"]
        if type(receipt.get("repair")) is not bool:
            raise ValueError("Invalid pending repair flag")
        expected_after = self._repaired_bytes(manifest) if receipt["repair"] else before
        action = "repair" if journal["events"] else "relocate"
        if (after != expected_after or receipt != self._receipt(manifest, before, after, action, receipt["repair"])
                or current_raw not in (before, after)):
            raise ValueError("Pending migration does not match manifest or receipt")
        if journal["events"] and journal["events"][-1]["manifest_after_sha256"] != hashlib.sha256(before).hexdigest():
            raise ValueError("Pending migration breaks event history")

        source = self._path("live/hard")
        destination = self._path("archive/hard")
        if source.exists() and destination.exists():
            raise ValueError("Both live and archive hard directories exist")
        if source.exists():
            if action != "relocate":
                raise ValueError("Repair cannot move the hard directory again")
            self._validate_hard_directory(manifest, "live/hard")
            self._path("archive").mkdir(exist_ok=True)
            source.rename(destination)
        self._validate_hard_directory(manifest, "archive/hard")
        if current_raw != after:
            self._atomic_write("manifest.json", after)
        # The in-memory mapping changes only after the complete manifest is installed.
        self._manifest = _decode_json(after, "updated manifest")
        self.rows = self._manifest["rows"]
        journal["events"].append(receipt)
        journal["pending"] = None
        self._atomic_write("relocation.json", _json_bytes(journal))
        return receipt

    @staticmethod
    def _repaired_bytes(manifest):
        rows = [
            {**row, "path": f"archive/hard/{row['id']}.json"} if row["difficulty"] == "hard" else dict(row)
            for row in manifest["rows"]
        ]
        return _json_bytes({**manifest, "rows": rows})

    def relocate(self, repair: bool = False) -> dict:
        """Rename live/hard once, optionally repairing its mapping atomically.

        relocation.json stores stable receipts and a write-ahead pending event so
        interrupted calls can resume after either the rename or manifest replace.
        Calls must be serialized by the owner of this isolated store.
        """
        repair = True  # Reference task repair: always publish the migrated mapping.
        if type(repair) is not bool:
            raise ValueError("repair must be a boolean")
        self._check_marker()
        manifest, before = self._load_manifest()
        files, already_repaired = self._migration_files(manifest)
        journal = self._load_journal(files)
        if journal["pending"] is not None:
            receipt = self._complete_pending(journal, before)
            return self.relocate(repair=True) if repair and not receipt["repair"] else receipt

        source = self._path("live/hard")
        destination = self._path("archive/hard")
        if source.exists() and destination.exists():
            raise ValueError("Both live and archive hard directories exist")
        if journal["events"]:
            previous = journal["events"][-1]
            if source.exists() or previous["manifest_after_sha256"] != hashlib.sha256(before).hexdigest():
                raise ValueError("Store state conflicts with relocation history")
            if previous["repair"] != already_repaired:
                raise ValueError("Repair receipt conflicts with manifest mapping")
            self._validate_hard_directory(manifest, "archive/hard")
            if not repair or already_repaired:
                self._manifest, self.rows = manifest, manifest["rows"]
                return previous
            action = "repair"
        else:
            if destination.exists() or already_repaired:
                raise ValueError("Archived backing files lack a relocation journal")
            self._validate_hard_directory(manifest, "live/hard")
            action = "relocate"
        after = self._repaired_bytes(manifest) if repair else before
        receipt = self._receipt(manifest, before, after, action, repair)
        journal["pending"] = {
            "receipt": receipt, "manifest_before": before.decode("utf-8"),
            "manifest_after": after.decode("utf-8"),
        }
        self._atomic_write("relocation.json", _json_bytes(journal))
        return self._complete_pending(journal, before)
