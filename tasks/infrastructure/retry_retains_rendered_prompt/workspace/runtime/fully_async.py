from __future__ import annotations

import copy

from .sample import Sample


def clone_group_for_retry(group: list[Sample]) -> list[Sample]:
    retried: list[Sample] = []
    for sample in group:
        replay = copy.deepcopy(sample)
        replay.reset_for_retry()
        retried.append(replay)
    return retried


def reset_source_group_for_recycle(group: list[Sample]) -> list[Sample]:
    for sample in group:
        sample.reset_for_retry()
    return group
