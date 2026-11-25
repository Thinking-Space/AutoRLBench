"""Decompose the BF16 window's 8.31 ms/step into geometry / kernel / merge.

Diagnostic only.  Opt-in via ``QAT_OSCAR_WINDOW_DECOMPOSE``; default off.  Nothing
here is meant to ship -- two of the three modes deliberately compute the WRONG
answer, which is why every one of them prints a loud banner.

Why this exists
---------------
j104820 puts the window at **8.31 ms/step** of a 17.31 ms decode step (mode 2,
plen 8192, B=16, 28 layers, full decode CUDA graphs): arm B 17.313 minus arm D
(window off) 9.00.  Task #68 assumed that cost is the *second attention* plus the
eager ``logaddexp`` merge, and that merging both tiers into one split-K reduction
the way upstream SGLang does would recover most of it.

Two roofline bounds say that assumption needs checking first.  The window is
64 sink + 256 recent = 320 tokens:

* **FLOPs** -- 16 seqs x 12 heads x 320 pos x 128 dim x 2 (QK and PV) x 2
  flop/MAC x 28 layers = 0.90 GFLOP/step.  Even at a pessimistic 10 TFLOP/s
  effective that is **0.09 ms**.
* **Bandwidth** -- the staging arena read is 16 x 320 x 2 (K and V) x 2 kv-heads
  x 128 x 2 bytes x 28 layers = 147 MB/step.  At the 570 GB/s this kernel was
  measured to sustain (job 52268) that is **0.26 ms**.

8.31 ms is ~30x above both.  So the time is very unlikely to be in the attention
arithmetic or in a six-op softmax merge over ``[B, Hq, D]`` = 24k elements.  The
remaining candidate is the ~25 small eager tensor ops per layer that build
``all_row`` and ``valid`` -- gathers, comparisons, ``cat``, ``contiguous`` on
``[B, P, bs]`` = 16x20x16 -- which under CUDA-graph replay cost no dispatch but
still execute as ~700 tiny kernels per decode step across 28 layers.  At the few
microseconds each that a launch-bound kernel costs even inside a graph, that is
the right order of magnitude.

If that is where the time is, the port to prioritise is folding the geometry into
the bounds kernel (P2/P3 of the original plan), not deleting ``logaddexp``.  This
module decides which, for the price of one 40-minute job instead of two days of
kernel work.

The decomposition
-----------------
Run the standard bench at three settings and difference them::

    full      (no env)                 INT2 + geometry + window kernel + merge
    skip_tail QAT_OSCAR_WINDOW_DECOMPOSE=skip_tail
                                       INT2 + geometry           + merge
    window off  VLLM_OSCAR_SINK_TOKENS=0 VLLM_OSCAR_RECENT_TOKENS=0
                                       INT2                              (= 9.00)

    window kernel  = full      - skip_tail
    geometry+merge = skip_tail - window_off

``skip_tail`` returns the ``-inf`` LSE that the merge already handles as "this
tier contributed nothing", so the pipeline stays numerically well-defined; the
OUTPUT IS WRONG (it is the INT2 tier alone) and ``out_sha`` will equal the
window-off arm's.  That equality is the check that the mode did what it claims.

Geometry and merge come out lumped together, and that is enough: the merge is
``logaddexp`` plus five elementwise ops over ``[B, Hq, D]`` = 24k elements, which
cannot be milliseconds.  If ``skip_tail - window_off`` is large, it is the
geometry.
"""

from __future__ import annotations

import os

import torch

ENV_FLAG = "QAT_OSCAR_WINDOW_DECOMPOSE"

_applied = False


def mode() -> str:
    return os.environ.get(ENV_FLAG, "") or ""


def apply(verbose: bool = True) -> bool:
    """Install the requested decomposition mode.  Returns True if installed."""
    global _applied
    m = mode()
    if _applied or m in ("", "0", "off"):
        return False
    if m != "skip_tail":
        raise ValueError(f"{ENV_FLAG}={m!r} is not one of: skip_tail")
    try:
        import vllm.v1.attention.backends.oscar_attn as OA
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[oscar-decompose] not applied: {type(exc).__name__}: {exc}",
                  flush=True)
        return False

    Impl = OA.OscarAttentionImpl

    def _bf16_tail_skipped(self, query, layer, all_blk, valid):
        """Return the empty-tier answer: zero output, -inf LSE.

        The caller's `logaddexp` merge already treats -inf as "contributed
        nothing", so `out` collapses to the INT2 tier exactly.
        """
        B = query.shape[0]
        dev = query.device
        o2 = torch.zeros(B, self.num_heads, self.head_size,
                         dtype=torch.float32, device=dev)
        lse2 = torch.full((B, self.num_heads), float("-inf"),
                          dtype=torch.float32, device=dev)
        return o2, lse2

    Impl._bf16_tail = _bf16_tail_skipped
    _applied = True
    if verbose:
        print(
            "[oscar-decompose] skip_tail ON -- the BF16 window kernel is NOT "
            "run; its tier returns -inf. Geometry and the merge still run. "
            "THE OUTPUT IS WRONG (INT2 only) and out_sha must match the "
            "window-off arm; if it does not, this mode is not doing what it "
            "says. Diagnostic only.",
            flush=True,
        )
    return True


def maybe_apply(verbose: bool = True) -> bool:
    return apply(verbose=verbose)
