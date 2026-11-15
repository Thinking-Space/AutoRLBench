"""Replay-buffer handling for routed-expert / indexer-topk data (design doc's "R3"):
adaptive-dtype wire encode/decode, plus interval-based, first-writer-wins
accounting for overlapping slices across resumed calls.

Native sglang engine responses are always int32 on the wire (mirrors
``miles.rollout.generate_utils.generate_endpoint_utils._decode_topk_buffer``'s
shape convention, generalized to any dtype for internal EngineClient use).
Narrower dtypes are an internal-only optimization for boundaries EngineClient
fully controls; any buffer handed back to *unmodified* existing parsers must
stay int32 -- see ``client_sdk.render_generate_response``, which always
passes ``dtype_mode="int32"``.

Each resumed engine call returns a replay buffer covering the *entire*
sequence generated so far, not just the newly generated tail -- so
successive commits overlap. ``ReplayLedger`` is first-writer-wins: a later
commit only contributes the sub-range not yet covered by an earlier commit,
so the final materialized array is stitched from the earliest data available
for each position instead of being overwritten by later, redundant re-sends.

Dense models never call ``commit`` with a non-``None`` slice, so a ledger
that's never received one is a no-op (``is_active`` is ``False``,
``materialize`` returns ``None``) -- no separate dense/MoE flag needed.
"""

import numpy as np
import pybase64

from miles.rollout.engine_client.types import ReplaySlice

_DTYPE_BY_NAME = {"uint8": np.uint8, "uint16": np.uint16, "int32": np.int32}


def choose_dtype(array: np.ndarray, mode: str) -> np.dtype:
    if mode != "auto":
        return _DTYPE_BY_NAME[mode]
    max_value = int(array.max()) if array.size else 0
    if max_value < 2**8:
        return np.uint8
    if max_value < 2**16:
        return np.uint16
    return np.int32


def encode_replay_buffer(array: np.ndarray, *, dtype_mode: str = "auto") -> tuple[str, str]:
    dtype = choose_dtype(array, dtype_mode)
    encoded = pybase64.b64encode(np.ascontiguousarray(array, dtype=dtype)).decode("ascii")
    return encoded, dtype.__name__


def decode_replay_buffer(
    info: str, *, num_tokens: int, num_layers: int, topk: int, dtype_name: str = "int32"
) -> np.ndarray:
    """``topk=-1`` recovers the indexer topk dimension from the buffer length,
    matching ``_decode_topk_buffer``'s convention. Always returns int32,
    regardless of wire dtype, to preserve ``Sample``'s external contract."""
    dtype = _DTYPE_BY_NAME[dtype_name]
    x = np.frombuffer(pybase64.b64decode(info.encode("ascii")), dtype=dtype)
    if num_tokens <= 0:
        return np.empty((0, num_layers, max(0, topk)), dtype=np.int32)
    if topk == -1:
        topk = len(x) // (num_tokens * num_layers)
    return x.reshape(num_tokens, num_layers, topk).astype(np.int32, copy=False)


class ReplayGapError(Exception):
    """Raised by ``materialize`` when the committed slices leave an
    uncovered gap inside ``[0, total_len)``."""

    def __init__(self, missing_range: tuple[int, int]):
        self.missing_range = missing_range
        super().__init__(f"replay ledger has an uncovered gap at {missing_range}")


class ReplayLedger:
    def __init__(self):
        self._intervals: list[tuple[int, int, np.ndarray]] = []
        self._empty_array: np.ndarray | None = None

    @property
    def is_active(self) -> bool:
        return self._empty_array is not None

    def commit(self, replay: ReplaySlice | None) -> None:
        if replay is None:
            return
        if self._empty_array is None:
            self._empty_array = np.empty((0, *replay.array.shape[1:]), dtype=replay.array.dtype)
        for lo, hi in _subtract_covered(replay.start, replay.end, self._covered_ranges()):
            offset = lo - replay.start
            self._intervals.append((lo, hi, replay.array[offset : offset + (hi - lo)]))
        self._intervals.sort(key=lambda item: item[0])

    def _covered_ranges(self) -> list[tuple[int, int]]:
        return [(lo, hi) for lo, hi, _array in sorted(self._intervals, key=lambda item: item[0])]

    def materialize(self, total_len: int) -> np.ndarray | None:
        if not self.is_active:
            return None
        if total_len <= 0:
            return self._empty_array

        cursor = 0
        chunks = []
        for lo, hi, array in sorted(self._intervals, key=lambda item: item[0]):
            if lo > cursor:
                raise ReplayGapError((cursor, lo))
            if hi <= cursor:
                continue
            chunks.append(array[cursor - lo :])
            cursor = hi
        if cursor < total_len:
            raise ReplayGapError((cursor, total_len))

        return np.concatenate(chunks, axis=0)[:total_len]


def _subtract_covered(start: int, end: int, covered: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sub-ranges of ``[start, end)`` not already present in ``covered``."""
    remaining = [(start, end)]
    for covered_lo, covered_hi in covered:
        next_remaining = []
        for lo, hi in remaining:
            if covered_hi <= lo or covered_lo >= hi:
                next_remaining.append((lo, hi))
                continue
            if covered_lo > lo:
                next_remaining.append((lo, covered_lo))
            if covered_hi < hi:
                next_remaining.append((covered_hi, hi))
        remaining = next_remaining
    return remaining
