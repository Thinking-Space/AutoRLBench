"""Data types for the token-level partial-rollout EngineClient.

These are new, self-contained types -- they don't replace or alias
``miles.utils.types.Sample`` or ``miles.rollout.session.types.SessionRecord``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class ReplaySlice:
    """A routed-expert / indexer-topk replay buffer covering absolute edge
    indices ``[start, end)`` in the ``len(prompt+output)-1``-sized convention
    used by ``Sample.rollout_routed_experts`` / ``Sample.rollout_indexer_topk``.

    Per the engine's own semantics, every engine response's replay buffer
    covers the full prefix generated so far (not just the newly generated
    tail), so in practice ``start`` is always ``0`` for a buffer straight off
    the wire; ``ReplayLedger`` is what turns a sequence of these
    overlapping, full-prefix buffers into one gap-free array.
    """

    start: int
    end: int
    array: "np.ndarray"


@dataclass(frozen=True)
class GenerationSegment:
    """One uninterrupted stretch of generation between two abort points (or
    between the start of a turn and the first abort, or the last abort and
    completion).

    ``raw_response`` retains the engine response except for the redundant
    routed-experts base64 payload. Renderers reconstruct that field from
    ``replay_routed_experts`` and preserve the other single-segment fields.
    """

    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    finish_reason: str  # "abort" | "stop" | "length"
    weight_version: str | None = None
    replay_routed_experts: ReplaySlice | None = None
    replay_indexer_topk: ReplaySlice | None = None
    raw_response: dict | None = None
    # Wall-clock time of the engine round trip that produced this segment
    # (``EngineClient._call_once``); used to report ``partial/processing_time``.
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class TurnRecord:
    """The complete, immutable result of one ``EngineClient.call()`` -- i.e.
    one logical generation turn, regardless of how many abort/resume segments
    the engine needed internally. Callers only ever see a finished
    ``TurnRecord``; the segment boundaries never leak out of ``EngineClient``.
    """

    turn_index: int
    prompt_token_ids: tuple[int, ...]
    segments: tuple[GenerationSegment, ...] = field(default_factory=tuple)
    engine_id: str | None = None

    @property
    def output_token_ids(self) -> list[int]:
        ids: list[int] = []
        for segment in self.segments:
            ids.extend(segment.token_ids)
        return ids

    @property
    def output_logprobs(self) -> list[float]:
        logprobs: list[float] = []
        for segment in self.segments:
            logprobs.extend(segment.logprobs)
        return logprobs

    @property
    def weight_versions(self) -> list[str]:
        return [segment.weight_version for segment in self.segments if segment.weight_version is not None]

    @property
    def processing_time(self) -> float:
        """Total engine round-trip time across all segments (i.e. including every resend)."""
        return sum(segment.elapsed_seconds for segment in self.segments)

    @property
    def finish_reason(self) -> str:
        """Finish reason of the terminal segment; ``"abort"`` if the turn has no segments yet."""
        return self.segments[-1].finish_reason if self.segments else "abort"
