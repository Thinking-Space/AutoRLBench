"""Turn a pytest log into per-test statuses.

The grader reads pytest's short test summary: one line per test, status first, which is stable
across pytest versions and easy to scan. The per-test progress lines that `--verbose` prints
are formatted by the terminal reporter and vary with plugins, so they are not parsed here.
"""
from __future__ import annotations

STATUSES = ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")

SUMMARY_MARKER = "short test summary info"


def normalize_node_id(raw: str) -> str:
    """Drop the file part of a node id: `tests/test_x.py::TestA::test_b` -> `TestA.test_b`."""
    return ".".join(raw.split("::")[1:])


def parse_pytest_log(log: str | None) -> dict[str, str]:
    """Map test name -> status for every test named in the log's summary section."""
    statuses: dict[str, str] = {}
    if log is None or SUMMARY_MARKER not in log:
        return statuses
    body = log.split(SUMMARY_MARKER)[1].strip()
    for line in body.split("\n"):
        if "PASSED" in line:
            statuses[normalize_node_id(line)] = "PASSED"
        elif "FAILED" in line:
            statuses[normalize_node_id(line).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            try:
                name = normalize_node_id(line)
            except IndexError:
                name = line
            statuses[name.split(" - ")[0]] = "ERROR"
    return statuses
