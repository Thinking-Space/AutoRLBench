"""Match the node ids a corpus instance requires against the statuses a run reported.

Node ids in the corpus come from the upstream task bank; what a log reports depends on the
pytest version, on parametrisation and on which plugins the repository enables, so the two
spellings are not always identical. The matcher is the one place that bridges them.
"""
from __future__ import annotations

from typing import Mapping

from .log_parse import normalize_node_id


def lookup(statuses: Mapping[str, str], node_id: str) -> tuple[str | None, str | None]:
    """Resolve one required node id against the parsed statuses.

    Returns (key, status). `key` is the entry the node id resolved to, or None when the run
    said nothing about it.
    """
    name = normalize_node_id(node_id)
    if name in statuses:
        return name, statuses[name]
    # Spelling drift: the bank id and the reported id agree up to a suffix the reporter adds
    # (parametrisation ids, class qualification). Fall back to the first entry that contains it.
    for key in statuses:
        if name in key:
            return key, statuses[key]
    return None, None


def resolve_all(statuses: Mapping[str, str], node_ids) -> dict[str, str | None]:
    """Resolve every required node id; the value is None when the run did not report it."""
    return {node_id: lookup(statuses, node_id)[1] for node_id in node_ids}
