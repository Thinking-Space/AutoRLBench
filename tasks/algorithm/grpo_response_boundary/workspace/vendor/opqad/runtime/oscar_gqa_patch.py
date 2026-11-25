"""Wire the GQA-reuse INT2 decode kernel into the serving backend, in-process.

What it replaces
----------------
The stock ``_oscar_decode_stage1`` is launched on a ``(B, Hq, splits)`` grid and
each program computes ``kv_head = hid // KV_GROUP_SIZE``.  Every KV page is
therefore loaded, dequantized and un-packed ``kv_group_size`` times -- 6x on the
1.5B, 4x on the 4B -- because the query heads sharing a KV head are handled by
independent programs that cannot see each other's loads.

``opqad/kernels/oscar_decode_gqa.py`` regrids to ``(B, Hk, splits)`` and gives one
program the KV head plus all G query heads that share it, so the page is loaded
once and the scores for all G heads come out of the same registers.  Stage 2 is
untouched: the Mid_o split layout is identical, only who fills it changes.

Measured, every configuration correctness-gated against the stock launcher
before timing (job 55616):

    shape                stock              GQA                speedup
    4B   b160 ctx32k     20.80 ms (145 GB/s) 13.43 ms (225)     1.55x
    1.5B b160 ctx32k      7.18 ms (105)       4.81 ms (157)     1.49x
    4B   b16  ctx8k       0.570 ms (132)      0.394 ms (192)    1.45x
    1.5B b16  ctx8k       0.243 ms (78)       0.176 ms (107)    1.38x

That is 1.5x against a 4-6x bound on redundant traffic, and the gap is
understood rather than mysterious: removing the redundant loads made the kernel
compute-bound, and every winning configuration is at ``num_warps=1,
BLOCK_KV=2``, i.e. register-limited.  Job 52268 measured the ceiling directly --
GB/s x group was 558/571/581/586 for groups 1/2/4/8, so the kernel was pinned at
~570 GB/s (66% of roofline) on group-redundant traffic no matter the group size.

Why a monkeypatch and not an edit
---------------------------------
The vLLM install at /shared/*/bitdistiller/envs/vllm-0.19.1-test/ is shared with
other people's running jobs and is never edited.  Both decode call sites in
``oscar_attn`` -- ``_decode_attention`` and ``_decode_attention_windowed`` --
resolve ``oscar_decode_attention`` through the module global imported at its
line 47, so rebinding that one name reaches both.

Capture safety
--------------
The launcher reads only shapes and strides (host-side metadata) and allocates
through the caller-supplied buffers when it can; there is no device sync and no
data-dependent host branch, so it is legal inside a CUDA graph capture.  It has
to be: with the staging probe fixed, cudagraph is on for OSCAR evals.

Opt-in via ``VLLM_OSCAR_GQA_DECODE=1``.  Off by default because it changes
decode numerics in the last bits -- the arithmetic is the same operations in a
different order, so the fp32 accumulation order differs across the G heads --
and a run meant to match an already-finished one should not enable it.

Which kernel and which launch config (changed 2026-09-16, job 61231)
--------------------------------------------------------------------
Until 2026-09-16 this dispatched ``oscar_decode_attention_gqa`` at
``BLOCK_KV=2`` and passed no ``num_stages`` (launcher default 1).  It passed no
``max_num_kv_splits`` either, but that one was NOT left at the launcher's
default of 16: both call sites in ``oscar_attn`` pass
``max_num_kv_splits=self.max_num_kv_splits``, which is
``attention_config.tq_max_kv_splits_for_cuda_graph``, and that defaults to 32
(vllm/config/attention.py:34).  Nothing in this repo overrides it.  So the real
deployed configuration was **gqa / BLOCK_KV=2 / splits=32 / warps=1 / stages=1**.

Job 61231 first swept this at ``B=160``, and those numbers are void: **the
engine never reaches B=160.**  Evals set ``MAX_NUM_SEQS=32`` and the staging
probe confirms ``reqs=32``; the training rollout runs 16 (train_batch_size 64
over 4 GPUs).  B is a grid dimension for both kernels, so that is a different
occupancy regime, not a detail.  Job 61375 re-swept at the shapes that occur:

    ms, warps=1, stages=1
    shape                 stock    deployed     packed     packed
                                   gqa 32/2     32/4       128/4
    1.5B b32  ctx32k      1.625     1.136       1.246      0.974
    1.5B b16  ctx17k      0.489     0.362       0.321      0.298
    4B   b32  ctx32k      4.299     2.794       2.678      2.617
    4B   b16  ctx17k      1.165     0.845       0.736      0.757

Findings:

* ``splits`` is where the money is, and 32 is not enough at these batches.
  Going 32 -> 128 is worth 1.08-1.24x on its own.  The deployed value came from
  ``tq_max_kv_splits_for_cuda_graph`` and nobody had swept it at the real batch.
  ``splits`` is also the only grid dimension not tied to the workload, which is
  why it matters most exactly where B is small.
* ``packed`` at ``BLOCK_KV=4 / splits=128`` is within 3% of the per-shape
  optimum on all four, so one config serves all of them.  ``gqa`` has no such
  value (it wants 2 at ctx32k and 4 at ctx8k).
* Against stock, that config is **1.54-1.67x** on the kernel.
* ``num_stages`` moves nothing (<0.5%, noise).  Exposed for the next GPU.

Read this honestly, twice over:

1. The INT2 kernel is ~56% of attention time (P1b), so 1.6x on it is ~1.3x on
   attention, not 1.6x end to end.
2. **The one clean-looking in-situ A/B disagrees.** Jobs 61082/61083 measured
   the GQA kernel 1.23x SLOWER per token than stock at this exact shape.  That
   pair is confounded -- the two arms ran on different nodes (lj-gpu4 vs
   lj-gpu6) with different co-tenant jobs -- so it does not settle anything
   either, but it is a standing warning that this kernel's offline wins have
   twice failed to appear in situ.  Do not quote the offline ratio as a
   throughput result until a same-node sequential A/B reproduces it.

Defaults are therefore ``packed``, ``BLOCK_KV=4``, ``splits=128``, ``warps=1``,
``stages=1``.  To reproduce a pre-2026-09-16 run exactly, set
``VLLM_OSCAR_DECODE_KERNEL=gqa VLLM_OSCAR_GQA_BLOCK_KV=2
VLLM_OSCAR_GQA_SPLITS=32``.

The live self-check is ON by default (``VLLM_OSCAR_GQA_VERIFY=0`` disables it).
It costs one duplicated launch and compares the selected kernel against the
stock one on the first real decode batch that has work -- which is the only
place a shape the offline bench never produced can be caught.

"That has work" is load-bearing.  Rows whose INT2 middle is empty come back
NaN by construction: ``oscar_attn`` passes ``seq_eff = cut - sink_pages *
block_size``, which is <= 0 while a sequence still fits inside sink + recent, so
every split is empty and stage2 evaluates ``exp(-inf - -inf)``.  Production
expects that and drops those rows (``empty1`` -> lse ``-inf``, ``nan_to_num``).
The first version of this check compared them anyway and printed
``max|diff|=nan rel=nan`` (job 61398), then marked itself verified -- a check
that reports NaN and passes has verified nothing.  It now masks empty rows,
defers to a later batch if none have work, and treats a non-finite row that
*does* have work as a failure.
"""
from __future__ import annotations

import os

from opqad.utils.envreg import env_bool, env_int, env_optint

_applied = False

# How many DISPATCHES to keep trying the self-check on before giving up.
#
# A dispatch is one layer of one decode step, not one decode step: `_dispatch`
# replaces `oscar_decode_attention`, which every layer calls every step.  The
# old value of 256 was written as if it counted decode steps, so on a 28-layer
# model it gave up after ~9 of them -- job 62812's rollout started from 63-token
# prompts and could not possibly have reached sink+recent=320 by then, and every
# run logged "UNVERIFIED" for a kernel that was in fact fine.
#
# 8192 dispatches is ~290 decode steps at 28 layers, ~230 at 36.  The cost is
# two extra kernel launches per dispatch, and only while still deferring.
# The number itself now lives in opqad/utils/envreg.py as the default for
# VLLM_OSCAR_GQA_VERIFY_MAX_TRIES, so there is one place to read it from.


def enabled() -> bool:
    return env_bool("VLLM_OSCAR_GQA_DECODE")


def maybe_apply(verbose: bool = True) -> bool:
    """Rebind oscar_attn.oscar_decode_attention.  Returns True if installed.

    Safe to call when vLLM has no OSCAR backend (a k2v3 or BF16 arm): the import
    fails and this returns False rather than raising.
    """
    global _applied
    if _applied or not enabled():
        return False

    kernel = (os.environ.get("VLLM_OSCAR_DECODE_KERNEL", "").strip().lower()
              or "packed")
    if kernel not in ("gqa", "packed"):
        if verbose:
            print(f"[oscar-gqa] unknown VLLM_OSCAR_DECODE_KERNEL={kernel!r}; "
                  "expected 'gqa' or 'packed'. Not applied.", flush=True)
        return False

    try:
        import torch
        import vllm.v1.attention.backends.oscar_attn as OA

        if kernel == "packed":
            from opqad.kernels.oscar_decode_packed import (
                oscar_decode_attention_packed as decode_fn,
            )
        else:
            from opqad.kernels.oscar_decode_gqa import (
                oscar_decode_attention_gqa as decode_fn,
            )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[oscar-gqa] not applied: {type(exc).__name__}: {exc}", flush=True)
        return False

    stock = OA.oscar_decode_attention
    # Job 61231's winners.  BLOCK_KV=4 is within 1% of optimal on all four
    # benched shapes for `packed`; `gqa` has no such single value (it wants 2 at
    # b160/ctx32k and 4 at b16/ctx8k), so a run pinned to gqa should set it.
    # splits=32 beat 16 everywhere, but 32 is also what the call sites already
    # pass (attention_config.tq_max_kv_splits_for_cuda_graph); naming it here
    # makes it visible and overridable rather than inherited silently.
    # These go through opqad/utils/envreg.py.  They are DEFAULTED there -- launch
    # geometry is bitwise-free -- but a value that is SET and unparseable now
    # raises instead of reverting to the default: the helper this replaced
    # turned VLLM_OSCAR_GQA_SPLITS=32x into 128 without a word.
    block_kv = env_optint("VLLM_OSCAR_GQA_BLOCK_KV")
    if block_kv is None:
        block_kv = 4 if kernel == "packed" else 2
    num_warps = env_int("VLLM_OSCAR_GQA_NUM_WARPS")
    splits = env_int("VLLM_OSCAR_GQA_SPLITS")
    num_stages = env_int("VLLM_OSCAR_GQA_NUM_STAGES")
    launch = {"block_kv": block_kv, "num_warps": num_warps,
              "num_stages": num_stages, "max_num_kv_splits": splits}
    # Self-check ON by default: run both kernels and compare, once, on the first
    # real decode.  The offline bench gates four synthetic shapes; this is the
    # only thing that catches a shape or a block table a live run produces and
    # the bench never did.  One duplicated launch, once per process.
    verify = env_bool("VLLM_OSCAR_GQA_VERIFY")
    max_tries = env_int("VLLM_OSCAR_GQA_VERIFY_MAX_TRIES")
    state = {"verified": not verify, "calls": 0, "max_seq": 0}

    def _dispatch(q_rot, kv_cache, block_table, seq_lens, scale, **kw):
        Hq = q_rot.shape[1]
        Hk = kv_cache.shape[2]
        # MHA has nothing to share, and a group of 1 would pay BLOCK_G overhead
        # for no reuse.  Hand those straight back to the stock kernel.
        if Hk <= 0 or Hq // Hk <= 1:
            return stock(q_rot, kv_cache, block_table, seq_lens, scale, **kw)

        state["calls"] += 1
        # `launch` last, so our tuned values override anything the call site
        # passed.  Merging rather than splatting both matters: `oscar_attn` may
        # pass `max_num_kv_splits` itself, and `f(**kw, max_num_kv_splits=32)`
        # would then be a duplicate-keyword TypeError at the first decode.
        call_kw = {**kw, **launch}

        if not state["verified"] and not torch.cuda.is_current_stream_capturing():
            # Compare on THIS batch, not on a synthetic one: the point is to
            # catch a shape or a block table the bench never produced.  Own
            # buffers for both, so neither can scribble on the caller's.
            ref_kw = {k: v for k, v in kw.items()
                      if k not in ("mid_o_buf", "output_buf", "lse_buf")}
            ref = stock(q_rot, kv_cache, block_table, seq_lens, scale, **ref_kw)
            got = decode_fn(q_rot, kv_cache, block_table, seq_lens, scale,
                            **{**ref_kw, **launch})
            # Rows whose INT2 middle is empty are legitimately NaN and the
            # caller drops them.  `oscar_attn` passes seq_eff = cut - sink_pages
            # * block_size, which is <= 0 whenever a sequence still fits inside
            # sink + recent; every split is then empty, stage2 does
            # exp(-inf - -inf), and the row comes back NaN.  Production handles
            # this (`empty1` masks lse1 to -inf and `o1 = nan_to_num(o1)`), so
            # it is not a bug -- but comparing those rows makes max|diff| NaN,
            # which is what job 61398's first self-check printed.  A check that
            # reports NaN and then marks itself verified has verified nothing.
            # seq_lens can be a CUDA-graph-padded buffer longer than the live
            # batch; slice it, or the boolean index below is the wrong length.
            live = (seq_lens.reshape(-1)[: ref.shape[0]] > 0)
            if live.any():
                state["max_seq"] = max(
                    state["max_seq"],
                    int(seq_lens.reshape(-1)[: ref.shape[0]][live].max().item()),
                )
            finite = torch.isfinite(ref).all(dim=-1).all(dim=-1) & \
                torch.isfinite(got).all(dim=-1).all(dim=-1)
            rows = live & finite
            n_rows = int(rows.sum().item())
            if n_rows == 0:
                # Nothing to compare on this batch.  Do NOT mark verified --
                # try again on the next one, when some request has grown past
                # sink + recent.  Each retry costs two extra launches, so cap
                # it: a workload whose sequences never leave the BF16 window
                # never touches the INT2 kernel for real either, and should say
                # so once rather than pay the duplicate launch forever.
                if state["calls"] == 1:
                    print(f"[oscar-gqa] self-check ({kernel}) deferred: every "
                          f"row of the first decode batch (B={q_rot.shape[0]}) "
                          f"has an empty INT2 middle. Will retry.", flush=True)
                elif state["calls"] >= max_tries:
                    state["verified"] = True
                    print(f"[oscar-gqa] self-check ({kernel}) GIVING UP after "
                          f"{state['calls']} dispatches (~{state['calls'] // 32} "
                          f"decode steps) with no row past sink+recent; longest "
                          f"sequence seen was {state['max_seq']} tokens. The "
                          f"kernel is UNVERIFIED for this run -- raise "
                          f"VLLM_OSCAR_GQA_VERIFY_MAX_TRIES if the workload is "
                          f"simply slow to grow.",
                          flush=True)
                return decode_fn(q_rot, kv_cache, block_table, seq_lens,
                                 scale, **call_kw)
            d = (ref[rows] - got[rows]).abs()
            diff = d.max().item()
            denom = ref[rows].abs().max().item() or 1.0
            # An all-NaN kernel output on a row that HAS work is a failure, not
            # a deferral -- count it so it cannot hide behind the mask above.
            bad = int((live & ~finite).sum().item())
            state["verified"] = True
            print(f"[oscar-gqa] self-check ({kernel}) on live batch "
                  f"B={q_rot.shape[0]} Hq={Hq} Hk={Hk}: compared {n_rows} rows "
                  f"with work ({bad} non-finite), max|diff|={diff:.3e} "
                  f"rel={diff/denom:.3e}", flush=True)
            if bad or diff / denom > 1e-3:
                raise RuntimeError(
                    f"[oscar-gqa] self-check FAILED for kernel {kernel!r}: rel "
                    f"{diff/denom:.3e} > 1e-3 or {bad} non-finite rows with "
                    "work. Refusing to serve a decode kernel that disagrees "
                    "with the reference; unset VLLM_OSCAR_GQA_DECODE."
                )

        return decode_fn(q_rot, kv_cache, block_table, seq_lens, scale, **call_kw)

    OA.oscar_decode_attention = _dispatch
    _applied = True
    if verbose:
        print(f"[oscar-gqa] decode kernel={kernel} ON (BLOCK_KV={block_kv}, "
              f"splits={splits}, num_warps={num_warps}, stages={num_stages}, "
              f"verify={int(verify)}).  One program per KV head instead of per "
              f"query head; job 61231 measured 1.07-1.75x over the previous "
              f"default (gqa/BLOCK_KV=2/splits=16) on the four benched shapes.  "
              f"Falls back to the stock kernel for MHA.", flush=True)
    return True
