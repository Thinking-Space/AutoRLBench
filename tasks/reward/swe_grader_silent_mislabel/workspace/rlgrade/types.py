"""Shared record types for the repository-repair grading pipeline.

One *record* is one graded rollout: a task instance (a repository snapshot plus the synthetic
bug that was injected into it) together with the patch a policy produced for it. The grader
turns a record into a reward by running the instance's tests on the patched working tree.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


def _resolve(path: str, corpus_root: Path) -> Path:
    """Corpus paths are relative to the corpus directory; environments to the workspace root.

    Symlinks are deliberately not followed: a virtual environment's `bin/python` is a link to
    the base interpreter, and resolving it hands the run an interpreter without the
    environment's packages.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(os.path.normpath(corpus_root.parent / candidate))


@dataclass(frozen=True)
class Instance:
    """One repository-repair instance of the training corpus."""

    instance_id: str
    repo_key: str
    # Pristine snapshot of the repository at the mirror commit, checked into the corpus.
    repo_root: Path
    # Interpreter of the per-repository environment the corpus builder installed.
    env_python: Path
    # Diff that injects the synthetic bug; applied before the policy patch.
    bug_patch: str
    # Tests the bug breaks, and tests that must keep passing. Full pytest node ids.
    fail_to_pass: Sequence[str] = ()
    pass_to_pass: Sequence[str] = ()
    # Extra pytest arguments this repository's suite needs (xdist, ini overrides, ...).
    extra_args: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, row: Mapping[str, Any], corpus_root: Path) -> "Instance":
        return cls(
            instance_id=row["instance_id"],
            repo_key=row["repo_key"],
            repo_root=(corpus_root / row["repo_root"]).resolve(),
            env_python=_resolve(row["env_python"], corpus_root),
            bug_patch=(corpus_root / row["bug_patch"]).read_text(encoding="utf-8"),
            fail_to_pass=tuple(row.get("fail_to_pass", ())),
            pass_to_pass=tuple(row.get("pass_to_pass", ())),
            extra_args=tuple(row.get("extra_args", ())),
            metadata=dict(row.get("metadata", {})),
        )

    @property
    def required_tests(self) -> tuple[str, ...]:
        """Every node id the reward depends on, in a stable order and without duplicates."""
        seen: dict[str, None] = {}
        for node in tuple(self.fail_to_pass) + tuple(self.pass_to_pass):
            seen.setdefault(node, None)
        return tuple(seen)


@dataclass(frozen=True)
class Submission:
    """A policy rollout for one instance."""

    record_id: str
    instance_id: str
    patch: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RunOutput:
    """Raw result of one test command."""

    command: str
    exit_code: int
    log: str
    seconds: float
    timed_out: bool = False


@dataclass
class Grade:
    """What the grader decided about one record."""

    record_id: str
    instance_id: str
    reward: float
    label: str
    statuses: Mapping[str, str] = field(default_factory=dict)
    missing: Sequence[str] = ()
    failed: Sequence[str] = ()
    notes: Sequence[str] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "instance_id": self.instance_id,
            "reward": self.reward,
            "label": self.label,
            "missing": list(self.missing),
            "failed": list(self.failed),
            "notes": list(self.notes),
        }
