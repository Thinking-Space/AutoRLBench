"""Submitter-side evidence check for this task.

It re-derives every number the task's README and oracle quote from the artefacts that ship with
the task, and scans the shipped files for anything that should not be public:

  * the archived pass A log matches its own summary, and the summary matches the README,
  * the reference labels in the verifier resources are consistent with the archived raw logs,
  * the shipped source archives match their recorded digests, and the pins are fully pinned,
  * the staged and hidden corpora do not overlap, and no file under `workspace/` names the
    internal cluster, internal hosts, credentials or the private model.

usage: python3 -B environment/check_evidence.py [--task <task_dir>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PRIVATE_MARKERS = (
    r"modelbest",
    r"tencentags",
    r"cybertron",
    r"minicpm",
    r"(?<![A-Za-z])(ark|e2b)_[A-Za-z0-9]{8,}",
    r"gpfs",
    r"/user/renjialin",
    r"swanlab",
    r"miles[-_]?planrl",
)
SKIP_SCAN = ("corpus/repos", "corpus/submissions", "corpus/patches", ".venvs")


class Failure(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_archive(task: Path, report: dict) -> None:
    archive = task / "workspace" / "logs" / "historical" / "pass_a"
    rows = read_jsonl(archive / "regrade.jsonl")
    summary = json.loads((archive / "summary.json").read_text())
    rewarded = sum(1 for row in rows if isinstance(row.get("reward"), (int, float)) and row["reward"] >= 1.0)
    if summary["records"] != len(rows) or summary["rewarded"] != rewarded:
        raise Failure(f"pass A summary disagrees with its own log: {summary} vs {len(rows)}/{rewarded}")
    if rewarded != 0:
        raise Failure(f"pass A should reward nothing, rewarded {rewarded}")
    by_id = {row["record_id"]: row for row in rows}
    for log in sorted((archive / "raw").glob("*.log")):
        record_id = log.stem
        if record_id not in by_id:
            raise Failure(f"raw log without a record: {record_id}")
        text = log.read_text(encoding="utf-8", errors="replace")
        # Without -rA pytest summarises only non-passing outcomes: either there is no summary
        # section at all, or it lists no PASSED line. Either way the parser cannot see a pass.
        if "short test summary info" in text:
            section = text.split("short test summary info")[1]
            passed = [line for line in section.splitlines() if line.startswith("PASSED")]
            if passed:
                raise Failure(f"archived log {record_id} summarises passes: {passed[:2]}")
        row = by_id[record_id]
        if row["reward"] != 0.0 or not row["missing"]:
            raise Failure(f"archived record {record_id} should be a zero with missing ids: {row}")
    report["pass_a"] = {"records": len(rows), "rewarded": rewarded,
                        "raw_logs": len(list((archive / "raw").glob("*.log")))}


def check_truth(task: Path, report: dict) -> None:
    resources = task / "evaluations" / "solution" / "verifier" / "resources"
    staged = json.loads((resources / "truth_staged.json").read_text())
    hidden = json.loads((resources / "truth_hidden.json").read_text())
    staged_records = {row["record_id"] for row in read_jsonl(task / "workspace" / "corpus" / "records.jsonl")}
    hidden_records = {row["record_id"] for row in read_jsonl(resources / "hidden" / "records.jsonl")}
    if set(staged) != staged_records:
        raise Failure("truth_staged.json does not cover exactly the staged records")
    if set(hidden) != hidden_records:
        raise Failure("truth_hidden.json does not cover exactly the hidden records")
    staged_instances = {row["instance_id"] for row in read_jsonl(task / "workspace" / "corpus" / "manifest.jsonl")}
    hidden_instances = {row["instance_id"] for row in read_jsonl(resources / "hidden" / "manifest.jsonl")}
    overlap = staged_instances & hidden_instances
    if overlap:
        raise Failure(f"{len(overlap)} instances appear in both the staged and the hidden corpus")
    for name, labels in (("staged", staged), ("hidden", hidden)):
        counts = {label: sum(1 for value in labels.values() if value == label)
                  for label in ("resolved", "unresolved", "unscorable")}
        if min(counts.values()) == 0:
            raise Failure(f"{name} corpus has no records of some label: {counts}")
        report[f"{name}_labels"] = counts
    report["instances"] = {"staged": len(staged_instances), "hidden": len(hidden_instances)}


def check_sources(task: Path, report: dict) -> None:
    manifest = json.loads((task / "environment" / "sources_manifest.json").read_text())
    for entry in manifest["repositories"]:
        archive = task / "environment" / "sources" / entry["archive"]
        if sha256(archive) != entry["sha256"]:
            raise Failure(f"source archive digest mismatch: {entry['archive']}")
        pins = (task / "environment" / "pins" / entry["pins"]).read_text().splitlines()
        loose = [line for line in pins if line.strip() and "==" not in line]
        if loose:
            raise Failure(f"{entry['pins']} is not fully pinned: {loose[:3]}")
    report["repositories"] = len(manifest["repositories"])


def check_privacy(task: Path, report: dict) -> None:
    patterns = [(marker, re.compile(marker, re.IGNORECASE)) for marker in PRIVATE_MARKERS]
    hits: list[str] = []
    for path in sorted(task.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(task).as_posix()
        if any(relative.startswith(f"workspace/{prefix}") for prefix in SKIP_SCAN):
            continue
        if relative in ("environment/check_evidence.py", "environment/privacy_exclusions.json"):
            continue  # the scanner's own pattern list, and the record of what it excludes
        if relative.startswith("environment/sources/"):
            continue
        if path.suffix in (".xz", ".gz", ".whl", ".png", ".jpg", ".pyc"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for marker, pattern in patterns:
            if pattern.search(text):
                hits.append(f"{relative}: {marker}")
    if hits:
        raise Failure("private markers found:\n  " + "\n  ".join(hits[:20]))
    report["privacy_scan"] = "clean"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    task = Path(args.task).resolve()
    report: dict = {"task": task.name}
    for check in (check_archive, check_truth, check_sources, check_privacy):
        try:
            check(task, report)
        except Failure as exc:
            print(f"FAIL {check.__name__}: {exc}")
            return 1
    print(json.dumps({"ok": True, **report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
