"""Construction of the pytest command used to grade one record.

The corpus protocol is *fail-to-pass plus pass-to-pass files*: the grader does not run the whole
suite, it runs the test files that hold the instance's required node ids. Running whole suites
costs minutes per rollout on the larger repositories and the extra tests are not part of the
reward anyway.
"""
from __future__ import annotations

from typing import Sequence

from .types import Instance

# Flags shared by every repository. Quiet output, no colour codes, no tracebacks: the grader
# reads the per-test statuses, not the failure text.
BASE_FLAGS: tuple[str, ...] = (
    "--disable-warnings",
    "--color=no",
    "--tb=no",
    "--verbose",
)


def test_files(instance: Instance) -> tuple[str, ...]:
    """The test files that contain the instance's required node ids, in manifest order."""
    files: dict[str, None] = {}
    for node in instance.required_tests:
        files.setdefault(node.split("::")[0], None)
    return tuple(files)


def build_command(instance: Instance, files: Sequence[str] | None = None) -> list[str]:
    """argv of the grading run for one instance."""
    argv = [str(instance.env_python), "-m", "pytest"]
    argv.extend(BASE_FLAGS)
    argv.extend(instance.extra_args)
    argv.extend(files if files is not None else test_files(instance))
    return argv
