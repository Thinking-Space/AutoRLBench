"""Per-session observability counters owned by the session server.

One :class:`SessionMetrics` per live session, created by ``SessionRegistry`` and
updated by ``SessionCore.chat_completions`` on every completed turn.  The summary
rides out through ``GET /sessions/{id}``'s ``metadata["session_metrics"]``, lands
on ``Sample.metadata``, and is reduced across the rollout batch by
``miles.ray.rollout.metrics``.

Two things are counted here that no other layer can see:

- **Branches.**  A turn whose prompt does not extend the trajectory's accumulated
  token IDs restarts the pretokenized prefix, i.e. opens a new branch.  Linear
  sessions never branch (``merge_tokens`` concatenates onto the stored prefix),
  so this stays at 1; the black-box fan-out providers registered through
  ``--session-provider-paths`` branch whenever they re-render from scratch.
  Counting it from the prompt token IDs keeps the metric provider-agnostic.

- **Token drift.**  The subset of branches where the *text* carried over intact
  but the *tokens* did not: chat-template or formatting rules re-render the
  previous turn's input+output identically, yet tokenization lands differently,
  so the prefix can no longer be reused.  Drift is a template/tokenizer defect
  and costs a prefix-cache miss, whereas a content-change branch is the agent
  legitimately rewriting history — worth separating in the dashboard.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

import numpy as np

from miles.utils.metric_utils import compute_statistics, dict_add_prefix


class TurnContinuity(Enum):
    """How one turn's prompt relates to the trajectory it was built on."""

    # The prompt extends the accumulated token IDs (or there were none yet).
    CONTINUED = "continued"
    # Prefix broken while the text was preserved — a tokenization-only change.
    BRANCH_TOKEN_DRIFT = "branch_token_drift"
    # Prefix broken because the text itself changed (rollback, retry, re-render).
    BRANCH_CONTENT_CHANGE = "branch_content_change"


def classify_turn_continuity(
    previous_token_ids: Sequence[int],
    prompt_token_ids: Sequence[int],
    *,
    max_trim_tokens: int,
    decode: Callable[[Sequence[int]], str],
) -> TurnContinuity:
    """Classify a turn from its prompt token IDs alone.

    ``max_trim_tokens`` trailing tokens of *previous_token_ids* are excluded from
    the comparison for the same reason ``update_pretokenized_state`` excludes
    them: a stop token the model emitted may legitimately re-render as the next
    turn's delimiter.

    ``decode`` is only called on the branch path, so the common continued turn
    costs one list comparison.
    """
    check_len = len(previous_token_ids) - max_trim_tokens
    if check_len <= 0:
        return TurnContinuity.CONTINUED

    previous_head = list(previous_token_ids[:check_len])
    if list(prompt_token_ids[:check_len]) == previous_head:
        return TurnContinuity.CONTINUED

    if decode(prompt_token_ids).startswith(decode(previous_head)):
        return TurnContinuity.BRANCH_TOKEN_DRIFT
    return TurnContinuity.BRANCH_CONTENT_CHANGE


def count_tool_calls(assistant_message: dict[str, Any]) -> list[str]:
    """Tool names requested by one assistant message, in call order.

    Unnamed calls are bucketed under ``"unknown"`` rather than dropped so the
    per-turn call count stays honest.
    """
    tool_calls = assistant_message.get("tool_calls") or []
    names = []
    for tool_call in tool_calls:
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        names.append(str(name) if name else "unknown")
    return names


@dataclass
class SessionMetrics:
    """Mutable per-session counters.

    Mutated only through :meth:`record_turn`, under the caller's session lock;
    :meth:`summarize` is read-only.
    """

    session_id: str
    created_at: float = field(default_factory=time.time)
    turns: int = 0
    tool_calls: int = 0
    tool_call_turns: int = 0
    tool_call_names: Counter = field(default_factory=Counter)
    branch_counts: Counter = field(default_factory=Counter)

    def record_turn(self, *, continuity: TurnContinuity, tool_call_names: Sequence[str]) -> None:
        self.turns += 1
        if continuity is not TurnContinuity.CONTINUED:
            self.branch_counts[continuity] += 1
        if tool_call_names:
            self.tool_call_turns += 1
            self.tool_calls += len(tool_call_names)
            self.tool_call_names.update(tool_call_names)

    @property
    def branches(self) -> int:
        """Distinct prefix chains this session produced (0 before the first turn)."""
        if self.turns == 0:
            return 0
        return 1 + sum(self.branch_counts.values())

    def summarize(self, now: float | None = None) -> dict[str, Any]:
        """JSON-serializable snapshot for ``GET /sessions/{id}``.

        ``duration`` is the server-side session lifetime (create -> read), which
        brackets the agent run: the rollout side opens the session immediately
        before invoking the agent function and reads it immediately after.
        """
        return {
            "session_id": self.session_id,
            "duration": (now if now is not None else time.time()) - self.created_at,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "tool_call_turns": self.tool_call_turns,
            "tool_call_names": dict(self.tool_call_names),
            "branches": self.branches,
            "token_drift_branches": self.branch_counts[TurnContinuity.BRANCH_TOKEN_DRIFT],
            "content_change_branches": self.branch_counts[TurnContinuity.BRANCH_CONTENT_CHANGE],
        }


# Branch counts above this are pooled into one open-ended bucket so a single
# runaway session cannot spray unbounded keys into the tracking backends.
_BRANCH_BUCKET_CAP = 8


def _branch_distribution(branches: Sequence[int]) -> dict[str, float]:
    """Fraction of sessions sitting at each branch count.

    Fractions rather than raw counts, for the same reason ``zero_std/*_percentage``
    are: the shape stays comparable across rollout batch sizes.
    """
    overflow = f"{_BRANCH_BUCKET_CAP + 1}+"
    buckets = Counter(str(n) if n <= _BRANCH_BUCKET_CAP else overflow for n in branches)
    return {f"branches_frac/{bucket}": count / len(branches) for bucket, count in buckets.items()}


def _tool_name_distribution(summaries: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Calls per session for each tool name, so the tool mix is readable per step."""
    names: Counter = Counter()
    for summary in summaries:
        names.update(summary.get("tool_call_names") or {})
    return {f"tool_name/{name}": count / len(summaries) for name, count in names.items()}


def aggregate_session_summaries(summaries: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Reduce per-session summaries into one flat metric dict for one rollout step.

    Follows the rollout-side statistical convention: distributions go through
    ``compute_statistics`` (mean/median/max/min), indicator quantities are logged
    as their mean over sessions, and discrete distributions as per-value
    fractions. Callers prefix the result (``rollout/session/``).
    """
    if not summaries:
        return {}

    turns = [int(s.get("turns", 0)) for s in summaries]
    tool_calls = [int(s.get("tool_calls", 0)) for s in summaries]
    branches = [int(s.get("branches", 0)) for s in summaries]
    drift_branches = [int(s.get("token_drift_branches", 0)) for s in summaries]
    content_branches = [int(s.get("content_change_branches", 0)) for s in summaries]
    durations = [float(s.get("duration", 0.0)) for s in summaries]

    metrics: dict[str, float] = {"num_sessions": float(len(summaries))}
    metrics |= dict_add_prefix(compute_statistics(turns), "turns/")
    metrics |= dict_add_prefix(compute_statistics(tool_calls), "tool_calls/")
    metrics |= dict_add_prefix(compute_statistics(branches), "branches/")
    metrics |= dict_add_prefix(compute_statistics(durations), "duration/")
    metrics |= dict_add_prefix(compute_statistics(drift_branches), "token_drift_branches/")
    metrics |= _branch_distribution(branches)
    metrics |= _tool_name_distribution(summaries)

    total_turns = sum(turns)
    if total_turns:
        metrics["tool_calls_per_turn"] = sum(tool_calls) / total_turns
        metrics["tool_call_turn_ratio"] = sum(int(s.get("tool_call_turns", 0)) for s in summaries) / total_turns
    total_branches = sum(branches)
    if total_branches:
        metrics["token_drift_branch_ratio"] = sum(drift_branches) / total_branches
        metrics["content_change_branch_ratio"] = sum(content_branches) / total_branches
    metrics["multi_branch_session_ratio"] = np.mean([b > 1 for b in branches]).item()
    metrics["token_drift_session_ratio"] = np.mean([d > 0 for d in drift_branches]).item()
    metrics["token_drift_branches_total"] = float(sum(drift_branches))
    return metrics
