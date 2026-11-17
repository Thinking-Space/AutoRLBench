"""The grading pipeline: record in, reward out.

Each record is graded in a scratch checkout of its instance:

    pristine snapshot -> inject the instance bug -> apply the rollout patch ->
    restore the tests -> run the required test files -> parse -> reward

The scratch checkout is a git repository, so the bug and the patch can be applied and undone
with the same tools the rollout itself used, and `git status` tells the trainer what a rollout
touched.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .command import build_command, test_files
from .executor import run_command
from .log_parse import parse_pytest_log
from .reset import restore_tests
from .reward import grade_statuses
from .types import Grade, Instance, Submission


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def materialize(instance: Instance, scratch: Path) -> Path:
    """Copy the pristine snapshot into the scratch directory and inject the instance bug."""
    repo = scratch / instance.repo_key
    if repo.exists():
        shutil.rmtree(repo)
    shutil.copytree(instance.repo_root, repo, symlinks=True)
    _git(repo, "reset", "--hard", "HEAD")
    _git(repo, "clean", "-qfd")
    patch_path = scratch / "bug.patch"
    patch_path.write_text(instance.bug_patch, encoding="utf-8")
    applied = _git(repo, "apply", "--whitespace=nowarn", str(patch_path))
    if applied.returncode != 0:
        raise RuntimeError(f"{instance.instance_id}: bug patch did not apply: {applied.stdout[-400:]}")
    return repo


def apply_submission(repo: Path, submission: Submission, scratch: Path) -> list[str]:
    """Apply the rollout patch. Returns notes describing what happened."""
    if not submission.patch.strip():
        return ["empty-patch"]
    patch_path = scratch / "submission.patch"
    patch_path.write_text(submission.patch, encoding="utf-8")
    applied = _git(repo, "apply", "--whitespace=nowarn", str(patch_path))
    if applied.returncode != 0:
        applied = subprocess.run(
            ["patch", "-p1", "--forward", "--batch", "-i", str(patch_path)],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return [] if applied.returncode == 0 else ["patch-failed"]


def grade_record(instance: Instance, submission: Submission, scratch: Path, timeout_s: int = 900) -> Grade:
    scratch.mkdir(parents=True, exist_ok=True)
    repo = materialize(instance, scratch)
    notes = apply_submission(repo, submission, scratch)
    restore_tests(repo)
    run = run_command(build_command(instance, test_files(instance)), cwd=repo, timeout_s=timeout_s)
    (scratch / "pytest.log").write_text(run.log, encoding="utf-8")
    statuses = parse_pytest_log(run.log)
    grade = grade_statuses(instance, statuses, submission.record_id, notes=notes)
    grade.notes = list(grade.notes) + [f"exit={run.exit_code}", f"seconds={run.seconds:.1f}"]
    return grade
