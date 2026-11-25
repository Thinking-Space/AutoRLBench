"""Replace OSCAR's per-token ``torch.quantile`` clip threshold with an order statistic.

The cost
--------
``OscarAttentionImpl._rotate_clip`` (oscar_attn.py:546) computes the clip
threshold as::

    thr = torch.quantile(x_rot.abs(), clip_ratio, dim=-1, keepdim=True)

over the head dimension (D=128), for **K and V, on every layer, for every token
stored** -- so twice per attention call, 2 x L times per decoded token.

``torch.quantile`` is a general routine: it validates, reshapes/transposes the
reduced dim to the end, runs a full ``sort`` over it, builds a rank tensor,
``gather``s the two bracketing order statistics and ``lerp``s them.  On a
``[16, 2, 128]`` decode tensor that is all dispatch and no arithmetic.  Job
104106's profile of a 1.5B at B=16 / 8192 ctx measured:

    aten::quantile   8064 calls   168 ms self CPU   1.732 s CPU total (11.53%)
                                  214.7 us per call   356 ms CUDA

1.732 s over 128 decode steps is **13.5 ms of the 71 ms step -- 19%** -- for a
reduction over 128 numbers.  The step is CPU-bound by 4.3x (self-CPU 117.3
ms/step vs 27.3 ms/step of CUDA), so this is time the GPU spends idle.

The replacement
---------------
``clip_ratio`` and ``D`` are fixed for a run, so the rank is a compile-time
constant and the whole thing is one ``topk`` plus one ``lerp``:

    v   = clip_ratio * (n - 1)      # in float32, see below
    lo  = floor(v);  frac = v - lo
    thr = lerp(sorted_asc[lo], sorted_asc[lo + 1], frac)

``topk(k = n - lo, largest=True, sorted=True)`` puts ``sorted_asc[lo]`` at
``[..., k-1]`` and ``sorted_asc[lo+1]`` at ``[..., k-2]``; for ratios near 0 the
ascending direction is cheaper and is used instead.  At the production ratios
k is 7 (0.96) and 12 (0.92), i.e. a partial selection over 128 rather than a
full sort.

Bitwise identity, and the one subtlety
--------------------------------------
This must not change a single bit of any run's output, and it does not -- but
only because the rank is computed the way ATen computes it.  ``quantile_impl``
builds ``ranks = q * (size - 1)`` as a tensor of the *input* dtype, so the
multiply rounds in **float32**, not in double::

    float32:  0.96 * 127 -> frac 0.9199981689453125
    float64:  0.96 * 127 -> frac 0.9200000000000017

Using the double value instead makes every threshold wrong in the last two
digits (relative 2.3e-7) -- close enough to pass an ``allclose`` and wrong
enough to fail the gate this change is held to.  ``_rank`` below does the
multiply in float32 on the CPU, once per ``(ratio, n)`` pair, and caches it;
nothing here touches the device to decide a rank, which would trade a sort for
a host sync and lose the point.

NaN is the one place the two routines genuinely differ, and it has to be put
back by hand.  ``torch.quantile`` propagates: if a reduced row contains any NaN
the whole threshold for that row is NaN (and the ``clamp`` downstream then
NaN-fills that head's vector).  An order statistic does not -- ``sort`` puts NaN
last, so index ``lo`` lands on a finite element and the threshold comes back
usable.  Left alone that would be a silent numerical divergence in exactly the
case where divergence is least likely to be noticed.

The repair is nearly free in the branch production uses: ``topk`` descending
orders NaN as greater than everything, so ``vals[..., 0]`` *is* the row maximum
and is NaN iff the row holds one.  Two elementwise kernels on ``[N, H, 1]``
against the ~6 dispatches plus a full sort being removed.  The ascending branch
pays a real ``isnan().any(-1)``; no production ratio takes it.

``tests/test_oscar_quantile_patch.py`` asserts ``torch.equal`` against
``torch.quantile`` rather than assuming any of this.

Always on when OSCAR is: there is no correct reason to sort 128 numbers to find
one of them.  ``QAT_OSCAR_FAST_QUANTILE=0`` restores the stock call for an A/B.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["enabled", "apply", "maybe_apply", "fast_quantile"]

_DONE = False
_applied = False

# (ratio, n, dtype) -> (lo, k, largest, frac).  Tiny and bounded: a run has one
# K ratio and one V ratio at one head size.
_RANKS: dict[tuple[float, int, Any], tuple[int, int, bool, float]] = {}


def enabled() -> bool:
    return os.environ.get("QAT_OSCAR_FAST_QUANTILE", "1").strip() != "0"


def _rank(ratio: float, n: int, dtype) -> tuple[int, int, bool, float]:
    """The two order statistics ``torch.quantile(x, ratio, dim=-1)`` interpolates.

    Returns ``(lo, k, largest, frac)``: take ``topk(x, k, largest=largest,
    sorted=True)``, and the bracketing pair is at the last two positions when
    ``largest`` and at ``[lo, lo+1]`` when not.
    """
    key = (ratio, n, dtype)
    hit = _RANKS.get(key)
    if hit is not None:
        return hit

    import torch  # noqa: PLC0415

    # float32 (the input dtype), NOT python double -- see the module docstring.
    r = torch.tensor(ratio, dtype=dtype) * float(n - 1)
    lo = int(torch.floor(r).item())
    frac = float((r - float(lo)).item())
    k_desc = n - lo          # sorted_asc[lo] is the k_desc'th largest
    k_asc = min(lo + 2, n)   # sorted_asc[lo] and [lo+1] are within the lo+2 smallest
    largest = k_desc <= k_asc
    out = (lo, k_desc if largest else k_asc, largest, frac)
    _RANKS[key] = out
    return out


def fast_quantile(x, ratio: float):
    """``torch.quantile(x, ratio, dim=-1, keepdim=True)``, bitwise, without the sort."""
    import torch  # noqa: PLC0415

    n = x.shape[-1]
    lo, k, largest, frac = _rank(ratio, n, x.dtype)
    vals = torch.topk(x, k, dim=-1, largest=largest, sorted=True).values
    if largest:
        a = vals.narrow(-1, k - 1, 1)                  # sorted_asc[lo]
        b = vals.narrow(-1, k - 2, 1) if k >= 2 else None  # sorted_asc[lo+1]
        # NaN sorts as greater than everything under topk, so the row maximum
        # is NaN exactly when the row holds one.  No reduction needed.
        nan_row = vals.narrow(-1, 0, 1)
    else:
        a = vals.narrow(-1, lo, 1)
        b = vals.narrow(-1, lo + 1, 1) if lo + 1 < k else None
        nan_row = None
    if b is None:
        # ratio == 1.0: lo is the last index, there is nothing above to
        # interpolate towards.  frac is 0 there, so this is the same value --
        # and `a` is the maximum, already NaN if the row has one.
        return a if largest else _propagate_nan(x, a)
    thr = torch.lerp(a, b, frac)
    if nan_row is None:
        return _propagate_nan(x, thr)
    # `torch.quantile` NaN-fills the whole reduced row; an order statistic does
    # not.  Reinstate it -- see the module docstring.
    return torch.where(nan_row.isnan(), nan_row, thr)


def _propagate_nan(x, thr):
    """The ascending branch's version: no free maximum, so reduce for it."""
    import torch  # noqa: PLC0415

    return torch.where(x.isnan().any(dim=-1, keepdim=True),
                       torch.full_like(thr, float("nan")), thr)


def apply(verbose: bool = True) -> bool:
    global _DONE, _applied
    if _DONE:
        return True
    if not enabled():
        return False

    import torch  # noqa: PLC0415
    from vllm.v1.attention.backends import oscar_attn as OA  # noqa: PLC0415

    Impl = OA.OscarAttentionImpl

    def _rotate_clip(self, x: torch.Tensor, R: torch.Tensor,
                     clip_ratio: float) -> torch.Tensor:
        """Rotate ``x`` (``[N, H, D]``) by ``R`` and optionally percentile-clip.

        Byte-for-byte the stock method with ``torch.quantile`` swapped out; the
        rotate, the ``.float()``, the ``abs`` and the ``clamp`` are unchanged.
        """
        x_rot = torch.matmul(x.float(), R)
        if clip_ratio > 0.0:
            thr = fast_quantile(x_rot.abs(), clip_ratio)
            x_rot = torch.clamp(x_rot, -thr, thr)
        return x_rot

    Impl._rotate_clip = _rotate_clip
    _DONE = True
    _applied = True
    if verbose:
        print(f"[oscar-quantile] pid={os.getpid()} _rotate_clip threshold via "
              "topk order statistic (bitwise-identical to torch.quantile)",
              flush=True)
    return True


def maybe_apply(verbose: bool = True) -> bool:
    return apply(verbose=verbose) if enabled() else False
