"""Opt-in: measure how much of OSCAR's BF16 sink/recent window actually survives.

Why this exists
---------------
INT2 KV is only tolerable because each sequence keeps its first ``sink_tokens``
(64) and last ``recent_tokens`` (256) in BF16.  Those live in a staging arena
allocated by ``_ensure_staging`` as **one fixed pool per layer, shared by every
concurrent request**, sized from ``staging_tokens`` (8192 by default).

``_oscar_staging_write_kernel`` addresses that pool as ``row = blk % n_rows`` --
a *direct-mapped* cache with one winner per row (``won = owner == blk``).  And
``_oscar_window_bounds_kernel`` only grants a request BF16 sink if **every**
sink slot is resident, then walks the recent tail back to the first unstaged
slot.  So a request needs all ~21 of its pages (block=16) resident at once:

    concurrency   working set    arena rows   outcome
    16 (rollout)   336 blocks       512       mostly resident
    160 (eval)    3360 blocks       512       ~15% capacity + conflict misses

When it does not fit, the code "degrades gracefully to the INT2 copy" --
gracefully in the sense of not crashing, and **with no log line at all**.  The
precision protection can therefore vanish completely without anything in the
output changing.  That is what this probe fixes.

What it reports
---------------
Per sampled decode step, for whichever layer the sample lands on:

    sink   fraction of requests that got their full BF16 sink
    tail   mean effective BF16 tail depth, against the nominal recent_tokens
    fit    arena rows vs the working set those requests actually need

How it measures
---------------
It wraps the launch of ``_oscar_window_bounds_kernel`` and reads that kernel's
own outputs -- ``_si`` (sink flag) and ``_cut`` (tail cut) -- rather than
reimplementing the residency rule in PyTorch.  A reimplementation would be a
second copy of a subtle rule, free to drift from the one that actually decides
what gets read in BF16; reading the real outputs cannot drift.

Sampling and capture safety
---------------------------
Reducing those tensors needs a device sync, which is illegal inside a capturing
stream -- and the cudagraph patch puts this very kernel under capture.  The
probe therefore skips any call made while the stream is capturing, and
otherwise samples one call in ``QAT_OSCAR_STAGING_PROBE`` (the env var is the
interval, not a boolean).  Unset or 0 disables it entirely.

Why that was not enough, and what the readback fixes
---------------------------------------------------
Skipping capturing calls silences the probe during capture.  What is easy to
miss is that it is equally silent during *replay*: a CUDA graph replays the
recorded kernels directly, so the Python wrapper below never executes again.
With ``QAT_OSCAR_FULL_CUDAGRAPH=1`` the only launches that ever reach
``_report`` are the eager ones -- vLLM's dummy profiling run, whose arena is
legitimately empty.  The probe then prints ``sink=0.0% tail=0.0%``, which reads
exactly like a total staging failure and is really a measurement of the warmup.
Jobs 53539-53542 were cancelled on that reading.

The buffers a captured graph writes to are fixed at capture time, so the fix is
to keep a reference to them and read them *after* a replay, outside any capture:
that is live data from the most recent decode step.  The probe does this on the
eager calls that keep arriving between replays -- only decode is captured, so
every prefill lands in the wrapper -- and those ``[graph B=...]`` lines are the
only reading that means anything when cudagraph is on.  The plain sampled lines
beside them come from the eager warmup and will say 0%.

Driving the readback from the driver process instead does not work: the probe
and the graphs live in the EngineCore worker, and ``collective_rpc`` with a
callable fails with "Object of type <class 'function'> is not serializable"
unless ``VLLM_ALLOW_INSECURE_SERIALIZATION=1``, which is not a flag worth
setting for a diagnostic.
"""
from __future__ import annotations

import os

ENV_FLAG = "QAT_OSCAR_STAGING_PROBE"

_applied = False
_calls = 0
_interval = 0


def interval() -> int:
    """Sampling interval from the environment; 0 means disabled."""
    try:
        return max(0, int(os.environ.get(ENV_FLAG, "0")))
    except ValueError:
        return 0


def enabled() -> bool:
    return interval() > 0


def _report(args, *, why: str = "") -> None:
    """Reduce one launch's outputs.  Positional layout is fixed by the call in
    ``_decode_attention_windowed``:

        owner, bt, seq, _cut, _si, <3 strides>, B, maxpg, bs, R, S_nb, S_eff, W
          0     1    2     3      4    5,6,7     8    9     10  11   12     13  14
    """
    import torch

    seq, cut, si = args[2], args[3], args[4]
    B, bs, R, S_nb, W = args[8], args[10], args[11], args[12], args[14]
    S_eff = args[13]
    if B <= 0:
        return

    # Same arithmetic _ensure_staging used to size the arena, so "needs" below
    # is the arena's own page accounting rather than a token-count estimate.
    tail_pages = (W + bs - 1) // bs + 1
    per_req = S_nb + tail_pages
    working_set = B * per_req

    sink_frac = (si > 0).float().mean().item()
    # seq - cut is the span the tail kernel will actually read in BF16; it is
    # capped at W because the window never extends further back than that.
    depth = (seq - cut).clamp(min=0, max=W).float()
    tail_mean = depth.mean().item()

    fit = (R / per_req) if per_req else float("inf")
    # seq is an INPUT buffer, so it settles the question a 0% reading cannot
    # answer on its own: is this a live view of a replay, or the stale
    # capture-time contents?  Under capture vLLM uses a fixed dummy seq_len; if
    # these numbers move with the run, the buffers are live and a 0% sink is a
    # real result rather than an artifact of reading a dead tensor.
    smin, smax = int(seq.min()), int(seq.max())
    if why and smax == 0:
        # Every sequence length zero on a graph readback means these tensors are
        # the capture-time dummies, not a live view of a replay: real decode
        # never has seq_len 0.  Print the fact instead of a 0% coverage figure
        # that would be read as "staging is dead".  Measured: this is what
        # actually happens, so the graph readback does NOT work and the
        # cudagraph question has to be answered from outputs, not instruments.
        print(
            f"[oscar-staging]{why} STALE -- seq is all zero, so these are the "
            f"capture-time buffers and not a live replay view.  No coverage "
            f"number can be read from them.",
            flush=True,
        )
        return
    # Which requests lost the sink -- the young ones or the old ones?  The
    # aggregate alone cannot tell "a freshly admitted request never staged its
    # sink" from "a long-running request had its region taken away", and the
    # two have different fixes.  Costs one masked min/max on a [B] tensor.
    nosink = (si <= 0) & (seq > S_eff)
    if bool(nosink.any()):
        ns = seq[nosink]
        nos = f"  nosink={int(nosink.sum())} seq=[{int(ns.min())},{int(ns.max())}]"
    else:
        nos = ""
    print(
        f"[oscar-staging]{why} reqs={B:4d}  sink={100*sink_frac:5.1f}% full  "
        f"tail={tail_mean:6.1f}/{W} ({100*tail_mean/max(W,1):5.1f}%)  "
        f"seq=[{smin},{smax}]  "
        f"arena={R} rows vs working set {working_set} "
        f"({per_req}/req -> fits {fit:.1f} reqs){nos}",
        flush=True,
    )


# Launch arguments seen while capturing, keyed by batch size.  The tensors are
# the very buffers the recorded graph writes into, so reading them after a
# replay yields that replay's values -- the only way to see inside a graph from
# Python.  Keyed by B because vLLM captures one graph per batch-size bucket.
_captured: dict = {}



class _ProbedKernel:
    """Transparent stand-in for the Triton JITFunction.

    Triton's launch syntax is ``kernel[grid](*args)``, so only ``__getitem__``
    needs intercepting; everything else forwards to the real kernel so that
    autotuning, warmup and attribute access behave identically.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getitem__(self, grid):
        launcher = self._inner[grid]

        def _run(*args, **kwargs):
            ret = launcher(*args, **kwargs)
            global _calls
            _calls += 1
            n = _interval
            if n:
                try:
                    import torch

                    # A sync here would be illegal under capture, and the
                    # cudagraph patch puts this kernel inside the graph.  Keep
                    # the arguments instead: the graph will write into these
                    # same buffers on every replay, and final_report() reads
                    # them once capture is over.  Without this the probe is
                    # blind for the entire run, because replay never re-enters
                    # this function.
                    if torch.cuda.is_current_stream_capturing():
                        B = args[8]
                        if B > 0:
                            _captured[int(B)] = args
                        return ret
                    if _calls % n == 0:
                        _report(args)
                        # This call is eager, which means no graph is replaying
                        # right now -- but the captured graphs' output buffers
                        # still hold whatever their last replay wrote.  Reading
                        # them here is the only way to see inside a graph from
                        # Python, and it needs no RPC and no serialization
                        # flag.  (An earlier attempt drove this from the driver
                        # via collective_rpc and died on "Object of type
                        # <class 'function'> is not serializable".)
                        #
                        # Eager calls do keep happening once capture is over:
                        # only decode is captured, so every prefill lands here.
                        for b in sorted(_captured):
                            _report(_captured[b], why=f" [graph B={b}]")
                except Exception as exc:  # noqa: BLE001
                    print(f"[oscar-staging] probe error: "
                          f"{type(exc).__name__}: {exc}", flush=True)
            return ret

        return _run

    def __getattr__(self, name):
        return getattr(self._inner, name)


def final_report() -> None:  # kept for manual use; the probe now self-reports
    """Read the captured graphs' live buffers, after generation has finished.

    Call this once the engine is idle.  With cudagraph off it prints nothing new
    (nothing was captured) and the per-sample lines above are the measurement.
    With cudagraph on these are the ONLY trustworthy lines: the sampled ones
    came from the eager dummy profiling run, whose arena is empty by
    construction and so reports 0% no matter what the real run did.
    """
    if not _captured:
        return
    print(f"[oscar-staging] post-replay readback of {len(_captured)} captured "
          f"graph(s); these reflect the last decode step each one ran.",
          flush=True)
    for b in sorted(_captured):
        _report(_captured[b], why=f" [graph B={b}]")



def apply(verbose: bool = True) -> bool:
    """Install the probe.  Returns True if it was installed.

    Safe to call when vLLM has no OSCAR backend (a k2v3 or BF16 arm): the
    import fails and this returns False rather than raising.
    """
    global _applied, _interval
    n = interval()
    if _applied or not n:
        return False
    try:
        import vllm.v1.attention.backends.oscar_attn as _oa
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[oscar-staging] not applied: {type(exc).__name__}: {exc}",
                  flush=True)
        return False

    _oa._oscar_window_bounds_kernel = _ProbedKernel(
        _oa._oscar_window_bounds_kernel
    )
    _interval = n
    _applied = True
    if verbose:
        print(f"[oscar-staging] coverage probe on, sampling 1 call in {n}. "
              f"Reports the fraction of requests holding a full BF16 sink and "
              f"the mean effective tail depth; both silently collapse when the "
              f"staging arena is oversubscribed.", flush=True)
    return True
