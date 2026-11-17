"""Restore the instance's tests before a record is graded.

The policy works in the same checkout the grader then tests. Nothing stops a rollout from
editing the tests it is supposed to make pass, so the grading step puts the test files back to
their pristine state first: the reward has to describe the repository's behaviour, not the
rollout's opinion of it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")


def iter_test_files(repo_root: Path):
    """Every test file of the checkout, as paths relative to the repository root."""
    for pattern in TEST_FILE_PATTERNS:
        for path in sorted(repo_root.rglob(pattern)):
            if ".git" in path.parts:
                continue
            yield path.relative_to(repo_root)


def restore_tests(repo_root: Path) -> list[str]:
    """Check the test files out of git again. Returns the files that were restored."""
    restored = []
    for rel in iter_test_files(repo_root):
        subprocess.run(
            f"git checkout -- {rel} 2>/dev/null",
            shell=True,
            cwd=str(repo_root),
        )
        restored.append(str(rel))
    return restored
