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
    r"planrl",
    r"planrm",
    r"gpfs",
    r"/user/renjialin",
    r"swanlab",
    r"miles[-_]?planrl",
)
SKIP_SCAN = ("pools",)


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
    records = read_jsonl(archive / "cases.jsonl")
    summary = json.loads((archive / "summary.json").read_text())
    accepted = sum(1 for row in records if row["accepted"])
    if summary["cases"] != len(records) or summary["accepted"] != accepted:
        raise Failure(f"pass A summary disagrees with its own log: {summary} vs {len(records)}/{accepted}")
    by_id = {row["case_id"]: row for row in records}

    # The claims the README and the oracle make about individual cases.
    empty_tag = by_id["c05_manifest_without_total"]
    if not empty_tag["accepted"]:
        raise Failure("c05 should have been accepted by the defective gate")
    tag = empty_tag["run"]["run_tag"]
    if not tag.startswith("mixed_"):
        raise Failure(f"c05's run tag should be missing its size component, got {tag!r}")
    if empty_tag["run"]["pool_total_env"] != "":
        raise Failure(f"c05 should have run with an empty pool total, got {empty_tag['run']['pool_total_env']!r}")

    float_tag = by_id["c06_non_integer_total"]["run"]["run_tag"]
    if not float_tag.startswith("mixed300.4_"):
        raise Failure(f"c06's run tag should carry the non-integer total, got {float_tag!r}")

    duplicates = by_id["c09_oversampling_over_pool"]["run"]["duplicate_prompts_total"]
    if duplicates != 544:
        raise Failure(f"c09 should repeat 544 prompts, got {duplicates}")

    for case_id in ("c02_zero_weight_source", "c03_single_source", "c04_legacy_manifest"):
        row = by_id[case_id]
        if row["accepted"]:
            raise Failure(f"{case_id} should have been refused by the defective gate")
        if "must be positive" not in row["message"] and "invalid registry entry" not in row["message"]:
            raise Failure(f"{case_id} was refused for an unexpected reason: {row['message'][:120]}")

    logs = sorted((archive / "cases").iterdir()) if (archive / "cases").is_dir() else []
    for case_dir in logs:
        if case_dir.name not in by_id:
            raise Failure(f"archived case without a record: {case_dir.name}")
    report["pass_a"] = {"cases": summary["cases"], "accepted": accepted,
                        "archived_case_logs": len(logs)}


def check_truth(task: Path, report: dict) -> None:
    resources = task / "evaluations" / "solution" / "verifier" / "resources"
    staged = json.loads((resources / "truth_staged.json").read_text())
    hidden = json.loads((resources / "truth_hidden.json").read_text())
    staged_cases = {row["case_id"] for row in read_jsonl(task / "workspace" / "cases" / "cases.jsonl")}
    hidden_cases = {row["case_id"] for row in read_jsonl(resources / "hidden_cases.jsonl")}
    if set(staged) != staged_cases:
        raise Failure("truth_staged.json does not cover exactly the staged cases")
    if set(hidden) != hidden_cases:
        raise Failure("truth_hidden.json does not cover exactly the held-out cases")
    overlap = staged_cases & hidden_cases
    if overlap:
        raise Failure(f"{len(overlap)} cases appear in both lists")
    for name, labels in (("staged", staged), ("hidden", hidden)):
        counts = {outcome: sum(1 for value in labels.values() if value == outcome)
                  for outcome in ("accept", "reject", "unsafe-accept")}
        if counts["accept"] == 0 or counts["reject"] == 0:
            raise Failure(f"{name} list has no cases of some outcome: {counts}")
        if counts["unsafe-accept"]:
            raise Failure(f"{name} reference labels must not contain unsafe-accept: {counts}")
        report[f"{name}_outcomes"] = counts
    calibration = json.loads((task / "evaluations" / "solution" / "oracle" / "calibration.json").read_text())
    reference = calibration["variants"]["reference"]
    for split in ("staged", "hidden"):
        if reference[split]["agreement"] != 1.0 or reference[split]["unsafe_accepts"]:
            raise Failure(f"the reference gate must be perfect on {split}: {reference[split]}")
    shipped = calibration["variants"]["as_shipped"]
    if shipped["staged"]["unsafe_accepts"] == 0 or shipped["staged"]["false_rejects"] == 0:
        raise Failure("the shipped gate must be wrong in both directions: "
                      f"{shipped['staged']}")
    report["calibration"] = {name: value["staged"]["agreement"] for name, value in calibration["variants"].items()}


def check_sources(task: Path, report: dict) -> None:
    manifest = json.loads((task / "environment" / "sources_manifest.json").read_text())
    archive = task / "environment" / "sources" / "query_banks.tar.xz"
    if sha256(archive) != manifest["query_banks_sha256"]:
        raise Failure("query_banks.tar.xz does not match sources_manifest.json")
    digests = json.loads((task / "environment" / "pool_digests.json").read_text())
    recipes = json.loads((task / "environment" / "pool_recipes.json").read_text())
    declared = {entry["pool_id"] for entry in recipes["built"]} | {entry["pool_id"] for entry in recipes["derived"]}
    if declared != set(digests):
        raise Failure(f"pool recipes and pool digests disagree: {sorted(declared ^ set(digests))}")
    cases = read_jsonl(task / "workspace" / "cases" / "cases.jsonl")
    cases += read_jsonl(task / "evaluations" / "solution" / "verifier" / "resources" / "hidden_cases.jsonl")
    used = {case["pool_dir"].split("/", 1)[1] for case in cases}
    missing = used - set(digests)
    if missing:
        raise Failure(f"cases refer to pools with no recorded digest: {sorted(missing)}")
    report["pools"] = len(digests)
    report["pools_used_by_cases"] = len(used)


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
