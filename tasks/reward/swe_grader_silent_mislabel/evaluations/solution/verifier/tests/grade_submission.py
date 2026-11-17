"""Grade a repaired reward pipeline.

Two passes decide the score:

* the system's rerun of `scripts/regrade_run.py` over the staged corpus, read from the fresh
  run's `experiment_logs/regrade.jsonl`, and
* a held-out pass this verifier drives itself: the same entry point, the repaired `rlgrade`
  package, and a corpus of instances and rollouts the workspace never contained. Both run the
  repositories' real test suites.

Both passes are compared with labels recorded by the reference pipeline. A pipeline passes when
it never calls an undecided or failing record `resolved`, and when it recovers the three labels
on both sets at the rates in resources/grader_spec.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[1]
RESOURCES = VERIFIER / "resources"
LABELS = ("resolved", "unresolved", "unscorable")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tree_digest(root: Path) -> str | None:
    """Digest of a snapshot's tracked content, independent of git metadata and build artefacts."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=str(root),
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if listed.returncode != 0:
        return None
    sha = hashlib.sha256()
    for rel in sorted(entry for entry in listed.stdout.split("\0") if entry):
        path = root / rel
        if not path.is_file():
            return None
        sha.update(rel.encode())
        sha.update(b"\0")
        sha.update(hashlib.sha256(path.read_bytes()).digest())
    return sha.hexdigest()


def integrity(workspace: Path, spec: dict) -> tuple[bool, dict]:
    """The pinned entry point, the corpus and the environments must be untouched."""
    details: dict[str, object] = {}
    pinned = json.loads((RESOURCES / "pinned_digests.json").read_text())
    mismatched = []
    for rel, expected in sorted(pinned.items()):
        staged = workspace / rel
        actual = digest(staged) if staged.is_file() else None
        if actual != expected:
            mismatched.append(rel)
    details["pinned_files"] = len(pinned)
    details["mismatched"] = mismatched[:20]

    snapshots, environments = {}, {}
    for repo_key, expected in sorted(spec["repo_tree_digests"].items()):
        repo = workspace / "corpus" / "repos" / repo_key
        actual = tree_digest(repo) if repo.is_dir() else None
        dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], cwd=str(repo),
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout.strip() \
            if repo.is_dir() else "missing"
        snapshots[repo_key] = {"digest_ok": actual == expected, "dirty": bool(dirty)}

        python = workspace / ".venvs" / repo_key / "bin" / "python"
        installed = subprocess.run(
            [str(python), "-m", "pip", "list", "--format=freeze", "--disable-pip-version-check"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ) if python.exists() else None
        present = {line.split("==")[0].strip().lower().replace("_", "-")
                   for line in (installed.stdout if installed and installed.returncode == 0 else "").splitlines()
                   if "==" in line}
        required = {line.split("==")[0].strip().lower().replace("_", "-")
                    for line in (RESOURCES / "pins" / f"{repo_key}.txt").read_text().splitlines()
                    if "==" in line} if (RESOURCES / "pins" / f"{repo_key}.txt").is_file() else set()
        environments[repo_key] = {
            "python": python.exists(),
            "missing_pins": sorted(required - present)[:5],
            "packages": len(present),
        }
    details["snapshots"] = snapshots
    details["environments"] = environments
    ok = (
        not mismatched
        and all(entry["digest_ok"] and not entry["dirty"] for entry in snapshots.values())
        and all(entry["python"] and not entry["missing_pins"] for entry in environments.values())
    )
    return ok, details


def score(rows: list[dict], truth: dict[str, str]) -> dict:
    """Compare one pass with the reference labels."""
    seen = {}
    for row in rows:
        record_id = row.get("record_id")
        if record_id in truth:
            seen[record_id] = row
    per_class = {label: {"total": 0, "correct": 0} for label in LABELS}
    false_positives, graded, agreed = [], 0, 0
    missing = [record_id for record_id in truth if record_id not in seen]
    for record_id, expected in sorted(truth.items()):
        row = seen.get(record_id)
        label = (row or {}).get("label")
        reward = (row or {}).get("reward")
        # The reward has to agree with the label it is reported next to.
        if label == "resolved" and reward != 1.0:
            label = f"inconsistent({label},{reward})"
        elif label == "unresolved" and reward != 0.0:
            label = f"inconsistent({label},{reward})"
        elif label == "unscorable" and reward is not None:
            label = f"inconsistent({label},{reward})"
        graded += 1
        per_class[expected]["total"] += 1
        if label == expected:
            agreed += 1
            per_class[expected]["correct"] += 1
        if label == "resolved" and expected != "resolved":
            false_positives.append(record_id)
    return {
        "records": graded,
        "agreement": round(agreed / graded, 4) if graded else 0.0,
        "false_positives": len(false_positives),
        "false_positive_examples": false_positives[:10],
        "missing_records": len(missing),
        "missing_examples": missing[:10],
        "recall": {
            label: round(per_class[label]["correct"] / per_class[label]["total"], 4)
            if per_class[label]["total"] else None
            for label in LABELS
        },
        "support": {label: per_class[label]["total"] for label in LABELS},
    }


def run_hidden_pass(workspace: Path, spec: dict, out_root: Path) -> tuple[list[dict], dict]:
    """Grade the held-out corpus with the repaired pipeline, running the real test suites."""
    staging = out_root / "hidden"
    corpus = staging / "corpus"
    shutil.copytree(RESOURCES / "hidden", corpus)
    # The held-out instances are graded against the workspace's own snapshots and environments,
    # which the integrity check has already pinned. Point each row at them by absolute path, so
    # the pass does not depend on how the repaired pipeline resolves relative corpus paths.
    manifest = corpus / "manifest.jsonl"
    rows = read_jsonl(manifest)
    for row in rows:
        row["repo_root"] = str(workspace / "corpus" / "repos" / row["repo_key"])
        row["env_python"] = str(workspace / ".venvs" / row["repo_key"] / "bin" / "python")
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    logs = out_root / "hidden_logs"
    command = [
        sys.executable, "-B", str(workspace / "scripts" / "regrade_run.py"),
        "--manifest", str(corpus / "manifest.jsonl"),
        "--records", str(corpus / "records.jsonl"),
        "--out", str(logs),
        "--scratch", str(out_root / "hidden_scratch"),
        "--workers", str(spec["hidden_workers"]),
        "--timeout-s", str(spec["hidden_timeout_s"]),
    ]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.time()
    completed = subprocess.run(
        command, cwd=str(workspace), env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=spec["hidden_pass_timeout_s"],
    )
    report = {
        "exit_code": completed.returncode,
        "seconds": round(time.time() - started, 1),
        "tail": completed.stdout[-1500:],
    }
    return read_jsonl(logs / "regrade.jsonl"), report


def main() -> int:
    workspace = Path(os.environ.get("AUTORLBENCH_WORKSPACE", "/workspace"))
    results = Path(os.environ.get("AUTORLBENCH_RESULTS", "/logs/artifacts"))
    spec = json.loads((RESOURCES / "grader_spec.json").read_text())
    report: dict[str, object] = {"task": "swe_grader_silent_mislabel"}

    integrity_ok, integrity_details = integrity(workspace, spec)
    report["integrity"] = {"passed": integrity_ok, **integrity_details}

    staged_rows = read_jsonl(results / "experiment_logs" / "regrade.jsonl")
    staged_truth = json.loads((RESOURCES / "truth_staged.json").read_text())
    report["staged_pass"] = score(staged_rows, staged_truth)

    hidden_report: dict = {}
    hidden_score: dict = {}
    if integrity_ok:
        with tempfile.TemporaryDirectory(prefix="hidden-pass-") as tmp:
            try:
                hidden_rows, hidden_report = run_hidden_pass(workspace, spec, Path(tmp))
            except subprocess.TimeoutExpired:
                hidden_rows, hidden_report = [], {"exit_code": None, "timed_out": True}
            hidden_truth = json.loads((RESOURCES / "truth_hidden.json").read_text())
            hidden_score = score(hidden_rows, hidden_truth)
    report["hidden_pass"] = {"run": hidden_report, **hidden_score}

    checks = {"integrity": integrity_ok}
    for name, measured in (("staged", report["staged_pass"]), ("hidden", hidden_score)):
        thresholds = spec["thresholds"][name]
        checks[f"{name}_agreement"] = bool(measured) and measured["agreement"] >= thresholds["agreement"]
        checks[f"{name}_no_false_positive"] = bool(measured) and measured["false_positives"] == 0
        for label in LABELS:
            floor = thresholds["recall"].get(label)
            if floor is None:
                continue
            recall = (measured.get("recall") or {}).get(label)
            checks[f"{name}_recall_{label}"] = recall is not None and recall >= floor
    report["checks"] = checks
    report["passed"] = all(checks.values())

    results.mkdir(parents=True, exist_ok=True)
    (results / "grading_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
