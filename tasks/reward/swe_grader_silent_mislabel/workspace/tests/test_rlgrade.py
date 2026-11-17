"""Unit tests of the grading pipeline.

They run without any repository checkout: the parser, the matcher and the reward rule are fed
recorded pytest output. Run them with any interpreter that has pytest:

    python3 -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlgrade.log_parse import parse_pytest_log  # noqa: E402
from rlgrade.match import lookup  # noqa: E402
from rlgrade.reward import grade_statuses  # noqa: E402
from rlgrade.types import Instance  # noqa: E402

# Captured from a run of one of the corpus repositories.
SUMMARY_LOG = """\
============================= test session starts ==============================
collected 3 items

tests/test_core.py::test_alpha PASSED                                    [ 33%]
tests/test_core.py::test_beta FAILED                                     [ 66%]
tests/test_extra.py::test_gamma PASSED                                   [100%]

=========================== short test summary info ============================
PASSED tests/test_core.py::test_alpha
FAILED tests/test_core.py::test_beta - AssertionError: 1 != 2
PASSED tests/test_extra.py::test_gamma
========================= 2 passed, 1 failed in 0.21s ==========================
"""


def instance(tmp_path: Path, required: list[str]) -> Instance:
    return Instance(
        instance_id="demo__repo.0000.bug__x",
        repo_key="demo__repo.0000",
        repo_root=tmp_path,
        env_python=Path(sys.executable),
        bug_patch="",
        fail_to_pass=tuple(required[:1]),
        pass_to_pass=tuple(required[1:]),
    )


def test_parses_every_reported_test():
    statuses = parse_pytest_log(SUMMARY_LOG)
    assert len(statuses) == 3


def test_failed_test_is_reported_as_failed():
    statuses = parse_pytest_log(SUMMARY_LOG)
    _, status = lookup(statuses, "tests/test_core.py::test_beta")
    assert status == "FAILED"


def test_reward_is_one_when_every_required_test_passes(tmp_path):
    statuses = parse_pytest_log(SUMMARY_LOG)
    target = instance(tmp_path, ["tests/test_core.py::test_alpha", "tests/test_extra.py::test_gamma"])
    grade = grade_statuses(target, statuses, record_id="r1")
    assert grade.reward == 1.0
    assert grade.label == "resolved"


def test_reward_is_zero_when_a_required_test_fails(tmp_path):
    statuses = parse_pytest_log(SUMMARY_LOG)
    target = instance(tmp_path, ["tests/test_core.py::test_beta", "tests/test_core.py::test_alpha"])
    grade = grade_statuses(target, statuses, record_id="r2")
    assert grade.reward == 0.0
    assert grade.label == "unresolved"


def test_a_required_test_the_run_never_mentions_is_not_rewarded(tmp_path):
    statuses = parse_pytest_log(SUMMARY_LOG)
    target = instance(tmp_path, ["tests/test_core.py::test_alpha", "tests/test_core.py::test_absent"])
    grade = grade_statuses(target, statuses, record_id="r3")
    assert grade.reward != 1.0
