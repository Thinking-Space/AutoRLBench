"""Turn per-test statuses into the scalar reward the trainer consumes.

The corpus protocol is all-or-nothing: a rollout is rewarded when every test the instance
requires passes -- the tests the bug broke and the tests that were already green. Partial
credit would reward patches that trade one broken test for another.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .match import resolve_all
from .types import Grade, Instance

RESOLVED = "resolved"
UNRESOLVED = "unresolved"


def grade_statuses(
    instance: Instance,
    statuses: Mapping[str, str],
    record_id: str,
    notes: Sequence[str] = (),
) -> Grade:
    resolved = resolve_all(statuses, instance.required_tests)
    failed = [node for node, status in resolved.items() if status is not None and status != "PASSED"]
    missing = [node for node, status in resolved.items() if status is None]
    reward = 0.0 if (not statuses or failed or missing) else 1.0
    return Grade(
        record_id=record_id,
        instance_id=instance.instance_id,
        reward=reward,
        label=RESOLVED if reward else UNRESOLVED,
        statuses=dict(statuses),
        missing=missing,
        failed=failed,
        notes=list(notes),
    )
