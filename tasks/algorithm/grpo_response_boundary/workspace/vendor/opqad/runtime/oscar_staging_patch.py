"""Per-request BF16 staging for the OSCAR INT2 backend.

The divergence this fixes
-------------------------
OSCAR (arXiv:2605.17757) keeps the first S tokens and the last R tokens of
*each request* in BF16 and quantizes the middle; at (S,R)=(64,256) the paper
reports 0.24% extra BF16 KV memory at 128K context.  The window is per-request,
indexed by that request's own decode position.

The vLLM backend we run implements it as ONE SHARED ARENA across all concurrent
requests, direct-mapped on the global cache block id:

    row = blk % n_rows                      (oscar_attn.py:215, 273, 293, ...)
    won = owner[row, off] == blk            last-writer-wins

`n_rows` comes from a fixed `staging_tokens` (8192 default -> 512 rows).  At 160
seats the working set is ~3339 blocks, so the arena holds ~15% of it, and every
tail write of every request can evict any other request's sink page.  Measured
consequence, from the staging probe:

    160 seats, staging  8192   sink 0-17%   tail 10-19% of the nominal 256
     16 seats, staging  8192   sink  0.0%   tail 100%
     16 seats, staging 131072  sink 20-93%  tail 76-100%

Even 26x overprovisioning leaves sink coverage ragged, because block ids are
drawn from the whole KV pool: over a long run, fresh tail pages keep aliasing
onto sink rows, and a sink page -- written once at prefill and never rewritten
-- is destroyed permanently when that happens.  So this is not a tuning problem.
No value of `staging_tokens` makes a shared direct-mapped cache behave like a
per-request reservation.

The fix
-------
Give every request a private, contiguous region, split by ROLE:

    PPR  = sink_pages + tail_pages                      rows per request
    rows = (max_num_seqs + 1) * PPR                     +1 = trash region

    row_sink(g, pg) = g*PPR + pg                        pg < sink_pages
    row_tail(g, pg) = g*PPR + sink_pages + pg % tail_pages

Sink and tail regions are disjoint, so a tail write can never evict a sink page
-- the failure mode above becomes structurally impossible.  Within the tail
region the modulo is a ring over the last `tail_pages` pages, which is the
recent window's own semantics (the oldest page ages out), not a conflict miss.
Across requests there is no sharing at all, so there are no conflict misses
between requests either.

`g` is a region, not a batch slot
---------------------------------
The attention metadata carries no request identity -- only `block_table`,
`seq_lens`, `slot_mapping` and `query_start_loc`.  The first version of this
patch therefore used the batch slot `r` as the region index, and carried rows
across vLLM's renumbering with a migration pass.

That lost the sink, permanently, once per admission: 1.5B at 16k (job 57770)
100% -> 46.9%, Qwen3-8B at 8k (59330) 100% -> 62.5%, Qwen3-8B at 32k (59474)
100% -> 3.1% -- one seat's worth per newly admitted request, tail unaffected,
in every eval run, while the same code driven from a probe script kept 100%
and logged healthy migrations (jobs 60012, 60047, 60092, 60248).  Model,
sequence length, parallel sampling and the forked EngineCore were each ruled
out by experiment.

So the slot is simply not a safe key, and `_region_map` replaces it with one
that is: `block_table[r, 0]`, the request's first cache block, fixed for its
lifetime.  Regions are looked up by that id, allocated when a request has
none (clearing the region's owner tags first), and released when no request in
the batch claims them -- so a *recycled* block id cannot inherit the previous
owner's staged bytes either.  Whatever vLLM does with slot indices, the region
follows the request.

Scope
-----
Everything here is an in-process monkeypatch.  The shared vLLM install at
/shared/*/bitdistiller/envs/vllm-0.19.1-test is never edited -- other people's
jobs run out of it.

Only two Triton kernels genuinely need replacing (`_oscar_staging_write_kernel`
and `_oscar_window_bounds_kernel`): they are the only ones that both compute a
row and consult the owner tag.  The BF16 tail kernels receive `all_blk` solely
to derive a row and take validity from a separately computed mask, so they are
fed pre-translated rows and run unmodified.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import triton
import triton.language as tl

_PATCHED = False
_MIG_EVENTS = 0


def _probe_interval() -> int:
    """Sampling interval for the staging probes; 0 means off.

    `QAT_OSCAR_STAGING_PROBE` is an *interval*, and both call sites below used to
    test `os.environ.get(...)` for truthiness.  The string "0" is truthy, so the
    documented off switch -- which is what `qat_e2e.sh:575` defaults to and what
    every training launcher exports -- never turned anything off.  j105105 was
    still printing `[oscar-staging-pr] region map ... call #44000` with
    `QAT_OSCAR_STAGING_PROBE=0` set, and that print costs three device->host
    syncs (`assign.has.sum()`, `assign.got.sum()`, `(blk0 < 0).sum()`) each time
    it fires.  Same parse as `opqad/diagnostics/oscar_staging_probe.interval()`, which had
    it right; this is the second instance of an env-var off switch being
    ignored by a truthiness test (the first was `bench_oscar_decode.py`).
    """
    try:
        return max(0, int(os.environ.get("QAT_OSCAR_STAGING_PROBE", "0").strip() or 0))
    except ValueError:
        return 0


# --------------------------------------------------------------------------
# Per-forward scratch: compute the layer-invariant bookkeeping once, not L times
# --------------------------------------------------------------------------
# Both hot paths below run once per layer per decoded token, and most of what
# they compute does not depend on the layer at all.  Job 104106's profile of a
# 1.5B at B=16 charged the decode step 5849 CUDA launches -- 209 per layer --
# against 27.3 ms of actual device work, i.e. the step is CPU-bound by 4.3x.
# Roughly 40 of those 209 are the same small tensor ops recomputed with
# identical inputs on all 28 layers.
#
# The cache key has to be "this model forward" exactly: too wide and a stale
# block table is read, too narrow and nothing is saved.  `attn_metadata` is the
# obvious candidate and is the wrong one -- under CUDA-graph capture vLLM reuses
# persistent metadata buffers across steps, so caching on its identity would
# serve a previous step's geometry.  `ForwardContext` is built fresh by
# `set_forward_context()` around every model forward (a plain dataclass, no
# __slots__, so it takes an attribute), which is the scope wanted.  If it is
# unavailable for any reason the helpers fall back to recomputing, which is
# today's behaviour.
def _step_cache():
    """Scratch dict scoped to the current model forward, or None.

    Returning None makes every caller recompute, which is exactly the
    pre-2026-09-18 behaviour -- so `QAT_OSCAR_STEP_CACHE=0` is the A/B switch
    the bitwise correctness gate is run through.  Read per call rather than
    latched at import: a test has to be able to flip it between two decodes of
    the same engine, and one os.environ lookup is nothing against the ~40
    tensor ops it decides.
    """
    if os.environ.get("QAT_OSCAR_STEP_CACHE", "1").strip() == "0":
        return None
    try:
        from vllm.forward_context import get_forward_context  # noqa: PLC0415

        ctx = get_forward_context()
    except Exception:  # noqa: BLE001  -- no forward context (tests, probes)
        return None
    if ctx is None:
        return None
    cache = getattr(ctx, "_oscar_step_cache", None)
    if cache is None:
        cache = {}
        try:
            ctx._oscar_step_cache = cache
        except Exception:  # noqa: BLE001  -- frozen/slotted dataclass
            return None
    return cache


def _cached(cache, key, holder, build):
    """`build()` once per (forward, key); `holder` is kept alive for identity.

    The identity check is what makes an `id()`-based key safe: ids are recycled,
    but a stale entry can only be returned if the recycled object is also the
    one stored, and the stored reference prevents that recycling for as long as
    the entry lives.

    `QAT_OSCAR_STEP_CACHE_CHECK=1` turns every hit into a proof: rebuild, and
    compare the two bitwise.  Both builders are pure -- `_DecodeGeom` reads
    metadata, `_region_assign` reads metadata and the region table and returns
    tensors without touching either -- so calling them a second time is
    side-effect-free, and a mismatch means the cache served a stale or a
    wrong-layer value.  This is the correctness gate for the whole hoist, and it
    is exact rather than statistical; it costs a host sync per comparison, so it
    is for tests and for one verification run, not for production.
    """
    if cache is None:
        return build()
    hit = cache.get(key)
    if hit is not None and hit[0] is holder:
        if os.environ.get("QAT_OSCAR_STEP_CACHE_CHECK", "0").strip() == "1":
            _assert_cache_hit_is_exact(key, hit[1], build())
        return hit[1]
    val = build()
    cache[key] = (holder, val)
    return val


def _assert_cache_hit_is_exact(key, cached, fresh) -> None:
    for name in type(cached).__slots__:
        a = getattr(cached, name)
        b = getattr(fresh, name)
        if a is None and b is None:
            continue
        if a is None or b is None or a.shape != b.shape or not torch.equal(a, b):
            raise RuntimeError(
                f"[oscar-staging-pr] per-forward cache is stale: field {name!r} "
                f"of {key[0]} differs from a fresh computation. The layer-"
                f"invariance assumption this hoist rests on does not hold here; "
                f"set QAT_OSCAR_STEP_CACHE=0 and report this."
            )


# `apply_patch` binds the backend module into its closure; the helpers above are
# module-level so the tests can drive them without a vLLM engine.  Resolve
# `_oscar_arange` (a cache keyed by (n, device) -- going around it would defeat
# the point, and it is already the top entry of the rollout's py-spy profile)
# once, lazily, rather than importing vLLM at module import time.
_ARANGE = None
_ARANGE_FALLBACK: dict = {}


def _arange_fallback(n: int, dev):
    """Same contract as oscar_attn._oscar_arange, for use without vLLM."""
    key = (n, str(dev))
    t = _ARANGE_FALLBACK.get(key)
    if t is None:
        t = _ARANGE_FALLBACK[key] = torch.arange(n, device=dev)
    return t


def _arange(n: int, dev):
    global _ARANGE
    if _ARANGE is None:
        try:
            from vllm.v1.attention.backends.oscar_attn import (  # noqa: PLC0415
                _oscar_arange,
            )

            _ARANGE = _oscar_arange
        except Exception:  # noqa: BLE001
            _ARANGE = _arange_fallback
    return _ARANGE(n, dev)


class _DecodeGeom:
    """The part of `_decode_attention_windowed` that does not know about layers.

    Everything here is a function of `attn_metadata` (block table, seq lens) and
    the arena geometry, which is identical on every OSCAR layer of the model:
    `stage_block` is the KV cache block size, and `sink_pages`/`tail_pages`/
    `sink_eff`/`recent_tokens` come from the OscarConfig.  They are folded into
    the cache key anyway rather than asserted.
    """

    __slots__ = ("offs", "spg", "spos_ok", "pg", "pg_ok_u", "pg_m", "tblk_u",
                 "pos", "pos_lt_seq", "seq_minus_w", "gidx_base", "bt64")

    def __init__(self, bt, seq, B, bs, S_nb, TP, S_eff, W, maxpg, dev):
        self.offs = _arange(bs, dev)
        self.bt64 = bt.to(torch.int64)
        # Sink pages: fixed page indices, so the whole in-window mask is a
        # constant for the step -- only `sink_active` is per-layer.
        if S_nb > 0:
            self.spg = _arange(S_nb, dev).view(1, S_nb)
            spos = (self.spg * bs).unsqueeze(-1) + self.offs.view(1, 1, bs)
            self.spos_ok = spos < S_eff                       # [1, S_nb, bs]
        else:
            self.spg = None
            self.spos_ok = None
        # Tail pages: the last TP pages of each request.
        last_page = (seq - 1) // bs
        self.pg = ((last_page - (TP - 1)).unsqueeze(1)
                   + _arange(TP, dev).unsqueeze(0))       # [B, TP]
        self.pg_ok_u = (self.pg >= 0).unsqueeze(-1)
        self.pg_m = self.pg.clamp(min=0)
        self.tblk_u = torch.gather(
            self.bt64, 1, self.pg.clamp(0, maxpg - 1)).unsqueeze(-1)
        self.pos = (self.pg * bs).unsqueeze(-1) + self.offs.view(1, 1, bs)
        self.pos_lt_seq = self.pos < seq.view(B, 1, 1)
        self.seq_minus_w = seq - W
        self.gidx_base = _arange(maxpg, dev).unsqueeze(0)


class _Bounds:
    """`cut` and `si` from the window-bounds kernel -- also layer-invariant.

    The kernel reads exactly four things: `owner` (the layer's slot-owner tags),
    `bt`, `seq`, and `reg` (the layer's region map).  The last two are per-forward.
    The other two are per-LAYER tensors that nevertheless hold identical bytes on
    every layer at the point this runs, and the reason is the same one that makes
    the region map layer-invariant:

      * `_region_map_impl` writes the same `blk0` and clears the same `rows` on
        every layer, from the same block tables in the same order;
      * `_oscar_staging_write_kernel_pr` derives every owner tag it stores from
        `slot`/`pos`/`seq`/`req`/`reg` and the arena geometry.  It never reads
        `key` or `value` to decide a tag -- those only supply the staged data.

    So layer 0's `(cut, si)`, computed after layer 0's write, is bit-for-bit what
    layer 27 would compute after layer 27's write.  Caching it turns 28 Triton
    launches per decode step into one; j104106 measured that kernel at 79.7 us
    per call and 2.23 ms per step, which is 9.1% of all device time.

    What this does NOT assume: that the arena contents match across layers.  They
    do not -- each layer stages its own K/V.  Only the ownership tags match, and
    the tags are all this kernel reads.
    """

    __slots__ = ("cut", "si")

    def __init__(self, cut, si):
        self.cut = cut
        self.si = si


def _bounds(OA, owner, bt, seq, B, maxpg, bs, n_rows, S_nb, S_eff, W, ppr, TP,
            region_of, stage_slots, dev):
    cut = torch.empty(B, dtype=torch.int64, device=dev)
    si = torch.empty(B, dtype=torch.int64, device=dev)
    # Launched through the module attribute, not a local name, so that
    # oscar_staging_probe's _ProbedKernel wrapper still intercepts it.  Note the
    # probe samples every Nth LAUNCH, so hoisting cuts its sampling rate by the
    # layer count for a given interval -- the coverage it reports is unchanged,
    # it just reports less often.
    OA._oscar_window_bounds_kernel[(B,)](
        owner, bt, seq, cut, si,
        owner.stride(0), owner.stride(1), bt.stride(0),
        B, maxpg, bs, n_rows, S_nb, S_eff, W,
        ppr, S_nb, TP,
        region_of, stage_slots,
        BLOCK_BS=triton.next_power_of_2(bs),
    )
    return _Bounds(cut, si)


def _decode_geom(attn_metadata, bt, seq, B, bs, S_nb, TP, S_eff, W, maxpg, dev):
    key = ("geom", id(attn_metadata), B, bs, S_nb, TP, S_eff, W, maxpg)
    return _cached(
        _step_cache(), key, attn_metadata,
        lambda: _DecodeGeom(bt, seq, B, bs, S_nb, TP, S_eff, W, maxpg, dev),
    )


# --------------------------------------------------------------------------
# Row arithmetic, shared by the kernels and the torch paths.  Kept in one place
# because six call sites have to agree on it exactly.
# --------------------------------------------------------------------------
@triton.jit
def _row_sink(r, pg, PPR):
    return r * PPR + pg


@triton.jit
def _row_tail(r, pg, PPR, sink_pages, tail_pages):
    return r * PPR + sink_pages + (pg % tail_pages)


@triton.jit
def _oscar_staging_write_kernel_pr(
    key_ptr, val_ptr, stage_k_ptr, stage_v_ptr, owner_ptr,
    slot_ptr, pos_ptr,
    stride_kt, stride_kh, stride_vt, stride_vh,
    stride_sr, stride_so, stride_sh,
    stride_ot, stride_oo,
    seq_ptr, req_ptr, reg_ptr,
    n_tokens, n_heads, head_dim, block_sz,
    PPR, sink_pages, tail_pages,
    sink_eff, recent_tokens,
    BLOCK_D: tl.constexpr,
    PHASE: tl.constexpr,
):
    """One program per (token, kv-head): decide window membership and store.

    Differs from the stock kernel in two ways.

    1.  The row is derived from (arena region, page index within the request)
        rather than from `blk % n_rows`.  The region comes from `reg_ptr`,
        which `_region_map` keys on the request's first block id -- not on the
        batch slot, which vLLM is free to renumber underneath us.

    2.  A token that is BOTH in the sink and in the recent window is written to
        BOTH regions.  The stock kernel had one row per token so the question
        could not arise; here the reader looks in the sink region for positions
        below the sink cut and in the tail region above it, and for a short
        sequence the recent window reaches down into the sink pages.  Writing
        one row would leave the other reader finding no owner and silently
        truncating its window.

    The two-phase owner protocol is preserved unchanged: phase 0 writes tags,
    the kernel boundary is a device-wide barrier, phase 1 writes data only where
    the tag survived.  With per-request regions the only remaining contention is
    the tail ring wrapping onto its own oldest page, which is intended eviction.
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    if t >= n_tokens:
        return

    slot = tl.load(slot_ptr + t).to(tl.int64)
    if slot < 0:
        return
    pos = tl.load(pos_ptr + t).to(tl.int64)
    r = tl.load(req_ptr + t).to(tl.int64)
    seq = tl.load(seq_ptr + r).to(tl.int64)
    reg = tl.load(reg_ptr + r).to(tl.int64)

    in_recent = pos >= (seq - recent_tokens)
    in_sink = pos < sink_eff
    if not (in_recent or in_sink):
        return

    blk = slot // block_sz
    off = slot % block_sz
    pg = pos // block_sz

    row_s = _row_sink(reg, pg, PPR)
    row_t = _row_tail(reg, pg, PPR, sink_pages, tail_pages)
    do_sink = in_sink and (pg < sink_pages)

    if PHASE == 0:
        if do_sink:
            tl.store(owner_ptr + row_s * stride_ot + off * stride_oo, blk)
        if in_recent:
            tl.store(owner_ptr + row_t * stride_ot + off * stride_oo, blk)
        return

    d = tl.arange(0, BLOCK_D)
    mask = d < head_dim
    k = tl.load(key_ptr + t * stride_kt + h * stride_kh + d, mask=mask, other=0.0)
    # V gets its OWN strides.  The stock kernel indexes V with K's strides,
    # which holds for Qwen2 (k and v are both slices of the fused qkv tensor)
    # and silently reads the wrong rows on any model whose k is a fresh tensor
    # by the time it reaches attention -- Qwen3's per-head k_norm makes k
    # contiguous [N, Hk*D] while v stays a slice with row stride (Hq+2Hk)*D.
    # Measured on Qwen3-8B: the BF16 window then attends with the right
    # weights over garbage values, NMSE 1-10 against dense attention with
    # nothing quantized (jobs 59268/59279).  See oscar_v_stride_patch.py.
    v = tl.load(val_ptr + t * stride_vt + h * stride_vh + d, mask=mask, other=0.0)

    if do_sink:
        won = tl.load(owner_ptr + row_s * stride_ot + off * stride_oo) == blk
        if won:
            base = row_s * stride_sr + off * stride_so + h * stride_sh + d
            tl.store(stage_k_ptr + base, k.to(stage_k_ptr.dtype.element_ty), mask=mask)
            tl.store(stage_v_ptr + base, v.to(stage_v_ptr.dtype.element_ty), mask=mask)
    if in_recent:
        won = tl.load(owner_ptr + row_t * stride_ot + off * stride_oo) == blk
        if won:
            base = row_t * stride_sr + off * stride_so + h * stride_sh + d
            tl.store(stage_k_ptr + base, k.to(stage_k_ptr.dtype.element_ty), mask=mask)
            tl.store(stage_v_ptr + base, v.to(stage_v_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _oscar_window_bounds_kernel_pr(
    owner_ptr, bt_ptr, seq_ptr, cut_ptr, si_ptr,
    stride_ot, stride_oo, stride_bt,
    n_req, maxpg, block_sz, n_rows, S_nb, S_eff, W,
    PPR, sink_pages, tail_pages,
    reg_ptr, n_regions,
    BLOCK_BS: tl.constexpr,
):
    """One program per request: the BF16 tail cut and the sink flag.

    Semantics identical to the stock kernel --
      sink_active = seq > S_eff AND every sink slot is staged
      cut         = earliest p in [max(seq-W, si*bs), seq) with every slot from
                    p to seq-1 staged; seq if none
    -- with rows taken from the per-request regions instead of `blk % n_rows`.
    The sink loop reads the sink region, the tail walk reads the tail region,
    matching what the write kernel stored.

    `n_rows` is unused by the arithmetic and kept only to preserve the stock
    kernel's positional signature through index 14: opqad/diagnostics/oscar_staging_probe.py
    reads seq/cut/si/B/bs/R/S_nb/W out of the launch args by fixed position, and
    that probe is the only instrument we have for window coverage.  Dropping the
    argument would silently disable it -- which is precisely the class of bug
    this whole patch exists to fix.
    """
    r = tl.program_id(0)
    if r >= n_req:
        return
    seq = tl.load(seq_ptr + r).to(tl.int64)
    offs = tl.arange(0, BLOCK_BS)
    off_ok = offs < block_sz
    reg = tl.load(reg_ptr + r).to(tl.int64)
    if reg >= n_regions:
        # No region was available for this request (only possible if more
        # requests are live than the arena was sized for).  Report "nothing
        # staged" rather than reading the trash region.
        tl.store(si_ptr + r, 0)
        tl.store(cut_ptr + r, seq)
        return

    sink_ok = seq > S_eff
    p = 0
    while p < S_nb:
        blk = tl.load(bt_ptr + r * stride_bt + p).to(tl.int64)
        row = _row_sink(reg, p, PPR)
        own = tl.load(
            owner_ptr + row * stride_ot + offs * stride_oo,
            mask=off_ok, other=-1,
        )
        staged = (own == blk) | (~off_ok)
        sink_ok = sink_ok & (tl.sum(staged.to(tl.int32)) == BLOCK_BS)
        p += 1
    si = tl.where(sink_ok, S_nb, 0)
    tl.store(si_ptr + r, si)

    t0 = tl.maximum(seq - W, si * block_sz)
    cut = seq
    pos = seq - 1
    done = False
    while (pos >= t0) and (not done):
        pg = pos // block_sz
        blk = tl.load(
            bt_ptr + r * stride_bt + tl.minimum(pg, maxpg - 1)
        ).to(tl.int64)
        o = pos % block_sz
        row = _row_tail(reg, pg, PPR, sink_pages, tail_pages)
        own = tl.load(owner_ptr + row * stride_ot + o * stride_oo)
        if (own == blk) and (pg >= 0):
            cut = pos
            pos -= 1
        else:
            done = True
    tl.store(cut_ptr + r, cut)


# --------------------------------------------------------------------------
# Torch-side row arithmetic, mirroring the two triton helpers above.
# --------------------------------------------------------------------------
def _rows_sink(r, pg, ppr):
    return r * ppr + pg


def _rows_tail(r, pg, ppr, sink_pages, tail_pages):
    return r * ppr + sink_pages + (pg % tail_pages)


class _RegionAssign:
    """The region map for one forward, shared by every layer.

    Why this is legitimate to share
    -------------------------------
    The assignment is a pure function of `block_table[:B, 0]` and the region
    table `layer._oscar_region_blk0`.  The block table is metadata: identical on
    every layer.  The region table is per-layer *storage*, but not per-layer
    *state* -- every OSCAR layer starts it at all -1 in `_ensure_staging` and
    then advances it by the same function of the same block tables, in the same
    order, on every forward.  So all L copies hold the same bytes at all times,
    and computing the assignment once and writing the result into each layer's
    copy is exactly what the per-layer computation would have produced.

    That is an invariant, not a proof, so it is checkable rather than assumed.
    `QAT_OSCAR_STEP_CACHE_CHECK=1` rebuilds the assignment on every cache hit --
    i.e. from every other layer's own region table -- and raises on any bitwise
    difference; `QAT_OSCAR_STEP_CACHE=0` restores the per-layer computation
    outright.  See `_cached`.

    The invariant can only break if some OSCAR layer misses a forward that
    others see -- which would already corrupt the staging arena, since the
    write kernel and the bounds kernel both key on these regions.
    """

    __slots__ = ("region", "rows", "padded", "has", "got")

    def __init__(self, region, rows, padded, has, got):
        self.region = region
        self.rows = rows
        self.padded = padded
        # Kept only so the coverage probe can report `matched`/`allocated` from
        # any layer, not just the one that happened to build the assignment.
        self.has = has
        self.got = got


def _region_assign(bt, blk0, S: int, B: int, ppr: int):
    """Everything in the region map that does not depend on the layer."""
    cur = bt[:B, 0].to(torch.int64)                      # [B]

    match = blk0.view(1, S) == cur.view(B, 1)            # [B, S]
    has = match.any(dim=1)                               # [B]
    reg = match.to(torch.uint8).argmax(dim=1).to(torch.int64)
    claimed = match.any(dim=0)                           # [S]
    free = ~claimed
    need = ~has

    # Pair the k-th request needing a region with the k-th free region.
    # cumsum-as-rank keeps this shape-static; the pairing is a bijection,
    # so no two requests can be handed the same region.
    free_rank = torch.cumsum(free.to(torch.int64), 0) - 1        # [S]
    need_rank = torch.cumsum(need.to(torch.int64), 0) - 1        # [B]
    alloc = (
        need.view(B, 1)
        & free.view(1, S)
        & (free_rank.view(1, S) == need_rank.view(B, 1))
    )
    got = alloc.any(dim=1)
    reg_alloc = alloc.to(torch.uint8).argmax(dim=1).to(torch.int64)
    trash = torch.full_like(reg, S)
    region = torch.where(has, reg, torch.where(got, reg_alloc, trash))

    # The committed table: release unclaimed regions, tag allocated ones.
    # The scatter sends every non-allocating request to the pad slot S, so
    # the duplicates it creates all write the same -1 to the same cell.
    padded = torch.cat(
        [torch.where(claimed, blk0, torch.full_like(blk0, -1)),
         torch.full((1,), -1, dtype=torch.int64, device=blk0.device)]
    )
    padded.index_copy_(
        0,
        torch.where(got, region, trash),
        torch.where(got, cur, torch.full_like(cur, -1)),
    )

    # Owner rows of freshly allocated regions, to be cleared.  Non-allocating
    # requests aim at the trash region, which is never read.
    pg = torch.arange(ppr, device=cur.device, dtype=torch.int64)
    rows = (torch.where(got, region, trash).view(B, 1) * ppr
            + pg.view(1, ppr)).reshape(-1)

    return _RegionAssign(region, rows, padded, has, got)


def _region_map_impl(self, layer: Any, attn_metadata, n_reqs: int):
    """Map this step's batch slots to arena regions, by request identity.

    The first design keyed a region on the *batch slot*, and carried the
    rows over with a migration pass when vLLM renumbered a slot.  That
    never worked outside a test harness: every eval run -- 1.5B at 16k
    (57770), 1.5B at 8k (59047), Qwen3-8B at 8k (59330) and at 32k (59474)
    -- lost exactly one request's sink per admission and ended at 3-60%
    coverage, while the same patch stack driven from a probe script kept
    100% and logged healthy migrations.  Rather than keep guessing which
    vLLM slot transition the migration misses, this removes the dependency:
    a region belongs to a *request*, identified by `block_table[r, 0]`,
    which is fixed for that request's lifetime.  Whatever vLLM does with
    slot indices -- condense, swap for the decode/prefill reorder, admit
    into a hole -- the lookup follows the request.

    The map is (re)computed once per step, in `_staging_write`, from the
    full metadata, and cached in `layer._oscar_region_of` for the decode
    path (decodes occupy the first slots, so a prefix of the cache is
    exactly right).  Three rules:

      match     a live request keeps the region tagged with its blk0
      free      a region no request in this batch claims is released --
                keeping the tag would let a *recycled* block id match it
                and hand a new request someone else's staged bytes
      allocate  a request with no region takes a free one, and that
                region's owner tags are cleared first, so nothing stale
                can be mistaken for staged

    Every step is fixed-shape tensor arithmetic: no `nonzero`, no boolean
    indexing, no `.item()`, therefore no host sync (`_migrate_renumbered`
    needed one per layer per step) and it is legal under graph capture.
    A request that finds no region at all maps to the trash region
    `stage_slots`, which the reader treats as "not staged".

    Since 2026-09-18 the ~25 ops that decide the assignment run ONCE per
    forward rather than once per layer (see `_RegionAssign`); what remains
    here is the four-op commit into this layer's own tables.
    """
    bt = attn_metadata.block_table
    S = self.stage_slots
    B = min(n_reqs, bt.shape[0])
    if B <= 0:
        return None
    global _MIG_EVENTS
    _MIG_EVENTS += 1
    n = _MIG_EVENTS
    blk0 = layer._oscar_region_blk0                      # [S]
    ppr = self.pages_per_req

    probe = _probe_interval()

    assign = _cached(
        _step_cache(), ("region", id(attn_metadata), B, S, ppr), attn_metadata,
        lambda: _region_assign(bt, blk0, S, B, ppr),
    )

    # Commit into THIS layer's tables.  Every tensor below was computed from a
    # region table identical to this one -- see _RegionAssign's docstring for
    # why, and for the two switches that check it.
    blk0.copy_(assign.padded[:S])
    # NOT `layer._oscar_slot_owner[assign.rows] = -1`.  That lowers to
    # `index_put_` with a python scalar, which ATen wraps in a CPU tensor and
    # copies it to the device -- and an unpinned H2D copy is illegal inside a
    # capturing stream:
    #     RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA
    #     graph capture unless the CPU tensor is pinned.
    # That is exactly what failed arm B of j104336, at this line, which means
    # per-request staging was never capture-legal -- despite being written to be
    # (fixed shapes, no nonzero, no .item(): necessary, not sufficient).
    # `index_fill_` passes the fill as a C++ Scalar by value into the kernel, so
    # there is no host tensor and nothing to copy.  Identical semantics here:
    # `rows` is 1-D and this is dim 0 of a [rows, bs] tensor, so each listed row
    # is filled with -1, and the duplicate trash-region indices that
    # non-allocating requests produce all write the same constant.
    layer._oscar_slot_owner.index_fill_(0, assign.rows, -1)
    region = assign.region
    layer._oscar_region_of[:B] = region

    # The cadence stays hardcoded at 2000 rather than following `probe`: the
    # eval launchers pass 200, and honouring it here would multiply their log
    # volume tenfold for no new information.  `probe` is consulted for on/off
    # only.
    if probe and (n <= 4 or n % 2000 == 0):
        print(f"[oscar-staging-pr] region map on "
              f"{getattr(layer, 'layer_name', '?')}: B={B} "
              f"matched={int(assign.has.sum())} "
              f"allocated={int(assign.got.sum())} "
              f"free_after={int((blk0 < 0).sum())}, call #{n}",
              flush=True)
    return region



def apply_patch() -> bool:
    """Swap the shared-arena staging for per-request regions. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True

    from vllm.config import get_current_vllm_config
    from vllm.v1.attention.backends import oscar_attn as OA

    Impl = OA.OscarAttentionImpl
    _orig_ensure = Impl._ensure_staging

    def _ensure_staging(self, layer: Any, kv_cache: torch.Tensor):
        if getattr(layer, "_oscar_stage_ready", False):
            return
        bs = kv_cache.shape[1]
        cfg = self.cfg
        self.stage_block = bs
        self.sink_eff = (cfg.sink_tokens // bs) * bs
        self.sink_pages = self.sink_eff // bs
        # The recent window can straddle one extra page, and the ring needs a
        # spare so the page being written is never the one being read.
        self.tail_pages = (cfg.recent_tokens + bs - 1) // bs + 1
        self.pages_per_req = self.sink_pages + self.tail_pages

        # Size from the engine's real concurrency.  A shared pool sized from a
        # token count was the whole defect; `staging_tokens` is deliberately
        # not consulted here.
        #
        # VLLM_OSCAR_STAGE_MAX_SEQS is the authority, not get_current_vllm_config():
        # that is a context-scoped global, and inside the EngineCore worker it
        # falls back to a DEFAULT VllmConfig -- jobs 52990-52993 sized for 256
        # slots on a 160-seat run because of exactly that.  Oversizing only wastes
        # memory, but undersizing would silently corrupt the row assignment, so
        # the launcher passes the number explicitly and the config is a fallback.
        max_seqs = 0
        env_seqs = os.environ.get("VLLM_OSCAR_STAGE_MAX_SEQS", "").strip()
        if env_seqs:
            max_seqs = int(env_seqs)
        if max_seqs <= 0:
            try:
                max_seqs = int(
                    get_current_vllm_config().scheduler_config.max_num_seqs
                )
                print("[oscar-staging-pr] WARNING: VLLM_OSCAR_STAGE_MAX_SEQS unset, "
                      f"falling back to vllm_config ({max_seqs}); this may be a "
                      "default rather than this run's max_num_seqs", flush=True)
            except Exception:  # noqa: BLE001
                max_seqs = 256
        self.stage_slots = max_seqs
        # One region per seat plus a trash region at index `max_seqs`: writes
        # for a request that has no region land there and are never read, which
        # is what lets the region map run without a single host sync (no
        # boolean-mask indexing, no nonzero, so it is also capture-safe).
        rows = (max_seqs + 1) * self.pages_per_req

        layer._oscar_stage_k = torch.zeros(
            rows, bs, self.num_kv_heads, self.head_size,
            dtype=torch.bfloat16, device=kv_cache.device,
        )
        layer._oscar_stage_v = torch.zeros_like(layer._oscar_stage_k)
        layer._oscar_slot_owner = torch.full(
            (rows, bs), -1, dtype=torch.int64, device=kv_cache.device
        )
        layer._oscar_stage_rows = rows
        # Region table: which request owns each region, by its first block id
        # (-1 = free), and the region each batch slot maps to this step.  See
        # _region_map.
        layer._oscar_region_blk0 = torch.full(
            (max_seqs,), -1, dtype=torch.int64, device=kv_cache.device
        )
        layer._oscar_region_of = torch.full(
            (max_seqs,), max_seqs, dtype=torch.int64, device=kv_cache.device
        )
        layer._oscar_stage_ready = True

        nbytes = (layer._oscar_stage_k.numel() * 2) * 2 + layer._oscar_slot_owner.numel() * 8
        if _probe_interval():
            print(
                f"[oscar-staging-pr] per-request arena: slots={max_seqs} "
                f"sink_pages={self.sink_pages} tail_pages={self.tail_pages} "
                f"rows={rows} block={bs} -> {nbytes/2**20:.1f} MiB per layer",
                flush=True,
            )

    def _staging_write(self, layer: Any, key, value, attn_metadata):
        """Per-request staging write.

        The `req` construction is lifted from opqad/runtime/oscar_cudagraph_patch.py and
        must stay that way: `torch.repeat_interleave(x, lengths_tensor)` syncs
        device->host to size its output, which is illegal inside a capturing
        stream.  This patch is applied AFTER the cudagraph patch and replaces the
        same method, so writing the naive form here silently re-broke FULL graph
        capture -- observed as `CUDA error: operation not permitted when stream
        is capturing` at engine start (jobs 52990-52993).
        """
        N = attn_metadata.num_actual_tokens
        if N == 0:
            return
        dev = key.device
        slot = attn_metadata.slot_mapping[:N].to(torch.int64)
        seq = attn_metadata.seq_lens.to(torch.int64)
        qsl = attn_metadata.query_start_loc.to(torch.int64)
        # num_reqs is seq.shape[0], not qsl.shape[0] - 1.
        n_reqs = seq.shape[0]
        # Regions are resolved once per step, here, and cached on the layer for
        # the decode reader.  This must run before the writes: allocation
        # clears the owner tags of a region it hands out.
        region = _region_map_impl(self, layer, attn_metadata, n_reqs)
        if region is None:
            return
        # Decide uniformity from CPU-side metadata: a .item() on a CUDA tensor is
        # the very sync that is illegal under capture.
        qsl_cpu = attn_metadata.query_start_loc_cpu
        if qsl_cpu is not None and qsl_cpu.shape[0] == n_reqs + 1:
            uniform = bool(((qsl_cpu[1:] - qsl_cpu[:-1]) == 1).all())
        else:
            uniform = N == n_reqs
        if uniform:
            req = OA._oscar_arange(n_reqs, dev)
        else:
            req = torch.repeat_interleave(
                OA._oscar_arange(n_reqs, dev), qsl[1:] - qsl[:-1]
            )
        if req.dtype != torch.int64:
            # _oscar_arange caches by (n, device) and other callers want int32.
            # The Triton kernel types req_ptr from the tensor passed at launch, so
            # a dtype change would force a JIT recompile every step.
            req = req.to(torch.int64)
        pos = seq[req] - qsl[req + 1] + OA._oscar_arange(N, dev)

        bs = self.stage_block
        sk, sv = layer._oscar_stage_k, layer._oscar_stage_v
        own = layer._oscar_slot_owner
        n_heads, head_dim = key.shape[1], key.shape[2]
        BLOCK_D = triton.next_power_of_2(head_dim)
        # The kernel indexes `reg` by request slot, so it has to cover every
        # slot the batch names, not only the ones the region map saw.
        reg = layer._oscar_region_of
        if region.shape[0] < n_reqs:
            reg = reg.clone()
            reg[region.shape[0]:n_reqs] = self.stage_slots
        for _phase in (0, 1):
            _oscar_staging_write_kernel_pr[(N, n_heads if _phase else 1)](
                key, value, sk, sv, own,
                slot, pos,
                key.stride(0), key.stride(1),
                value.stride(0), value.stride(1),
                sk.stride(0), sk.stride(1), sk.stride(2),
                own.stride(0), own.stride(1),
                seq, req, reg,
                N, n_heads, head_dim, bs,
                self.pages_per_req, self.sink_pages, self.tail_pages,
                self.sink_eff, self.cfg.recent_tokens,
                BLOCK_D=BLOCK_D, PHASE=_phase,
            )

    def _stage_splice(self, layer: Any, bt_row, cached_len: int, k_cached, v_cached):
        """Replace INT2-reconstructed prefix rows with their staged BF16 values.

        Needs the request's batch slot, which the stock signature does not carry.
        `_prefill_attention` calls this as `_stage_splice(layer, block_table[i],
        ...)`, and `block_table[i]` is a view whose storage offset is
        `block_table.storage_offset() + i * block_table.stride(0)`, so the slot
        is recoverable exactly -- no matching heuristic, no call counting (the
        call index is not `i`, since requests with `cached_len <= 0` are skipped).
        """
        bt = getattr(self, "_stage_splice_bt", None)
        if bt is None:
            return k_cached, v_cached
        r = (bt_row.storage_offset() - bt.storage_offset()) // bt.stride(0)
        if not (0 <= r < self.stage_slots):
            return k_cached, v_cached
        # Batch slot -> arena region, through the map this step's
        # `_staging_write` already committed.
        r = int(layer._oscar_region_of[r])
        if not (0 <= r < self.stage_slots):
            return k_cached, v_cached
        bs = self.stage_block
        ppr, sp, tp = self.pages_per_req, self.sink_pages, self.tail_pages
        dev = k_cached.device
        npg = (cached_len + bs - 1) // bs
        blk = bt_row[:npg].to(torch.int64)
        pg = torch.arange(npg, device=dev, dtype=torch.int64)
        # A prefix page is read from the sink region when it is a sink page and
        # from the tail region otherwise -- the same split the reader uses.
        rows = torch.where(
            pg < sp,
            _rows_sink(r, pg, ppr),
            _rows_tail(r, pg, ppr, sp, tp),
        )
        staged = layer._oscar_slot_owner[rows] == blk.unsqueeze(-1)  # [npg, bs]
        pos = (pg * bs).unsqueeze(-1) + torch.arange(bs, device=dev)
        staged = (staged & (pos < cached_len)).reshape(-1)[:cached_len]
        ks = layer._oscar_stage_k[rows].reshape(npg * bs, self.num_kv_heads, -1)
        vs = layer._oscar_stage_v[rows].reshape(npg * bs, self.num_kv_heads, -1)
        m = staged.view(-1, 1, 1)
        k_out = torch.where(m, ks[:cached_len].to(k_cached.dtype), k_cached)
        v_out = torch.where(m, vs[:cached_len].to(v_cached.dtype), v_cached)
        return k_out, v_out

    _orig_prefill = Impl._prefill_attention

    def _prefill_attention(self, query, key, value, kv_cache, attn_metadata, layer):
        """Unchanged, except that it hands _stage_splice the block table it is
        iterating so the batch slot can be recovered from the view offset."""
        self._stage_splice_bt = attn_metadata.block_table
        try:
            return _orig_prefill(self, query, key, value, kv_cache,
                                 attn_metadata, layer)
        finally:
            self._stage_splice_bt = None

    def _decode_attention_windowed(self, query, kv_cache, attn_metadata, layer):
        """Same structure as the stock method; rows come from the per-request
        regions, and the BF16 tail path is handed rows instead of block ids.

        The tail kernels use `all_blk` only to derive `blk % R`; validity comes
        from the separately computed `valid` mask.  Passing rows that are
        already in [0, R) makes `row % R == row`, so those kernels and
        `_bf16_tail_eager` run unmodified against the new layout.
        """
        B = query.shape[0]
        Hq = self.num_heads
        bs = self.stage_block
        dev = query.device
        owner = layer._oscar_slot_owner
        bt = attn_metadata.block_table
        seq = attn_metadata.seq_lens.to(torch.int64)
        maxpg = bt.shape[1]
        S_nb, TP, S_eff = self.sink_pages, self.tail_pages, self.sink_eff
        ppr = self.pages_per_req
        W = self.cfg.recent_tokens
        # Page indices, token positions and the block ids they name depend only
        # on the block table and the sequence lengths, so they are the same on
        # all L layers of this forward.  Built once; see _DecodeGeom.
        g = _decode_geom(attn_metadata, bt, seq, B, bs, S_nb, TP, S_eff, W,
                         maxpg, dev)
        offs = g.offs
        # Rows come from the region this request owns, not from its batch slot.
        # `_staging_write` committed the map earlier in this same forward, and
        # decodes occupy the leading slots, so the prefix is exactly this
        # batch's mapping.  Requests with no region point at the trash region,
        # which the bounds kernel reports as "nothing staged" and whose rows
        # exist in the arena, so the gathers below stay in bounds.
        reg_of = layer._oscar_region_of[:B].view(B, 1)
        rq = reg_of

        _b = _cached(
            _step_cache(),
            ("bounds", id(attn_metadata), B, S_nb, TP, ppr, maxpg, bs, S_eff, W),
            attn_metadata,
            lambda: _bounds(OA, owner, bt, seq, B, maxpg, bs,
                            layer._oscar_stage_rows, S_nb, S_eff, W, ppr, TP,
                            layer._oscar_region_of, self.stage_slots, dev),
        )
        cut, si = _b.cut, _b.si
        sink_active = si > 0

        # There is no `sblk`.  The stock method computed `bt[:, :S_nb]` here and
        # never read it: the BF16 tail kernels take `all_row` and `valid`, and
        # take validity from the mask, not from a block id.  It was two tensor
        # ops per layer per token for nothing.
        if S_nb > 0:
            srow = _rows_sink(rq, g.spg, ppr)
            # g.spos_ok is `spos < S_eff`, which is fixed for the step; only
            # `sink_active` is per-layer.
            s_valid = (sink_active.view(B, 1, 1) & g.spos_ok).expand(B, S_nb, bs)
        else:
            srow = torch.zeros(B, 0, dtype=torch.int64, device=dev)
            s_valid = torch.zeros(B, 0, bs, dtype=torch.bool, device=dev)

        trow = _rows_tail(rq, g.pg_m, ppr, S_nb, TP)
        town = owner[trow.unsqueeze(-1), offs.view(1, 1, bs)]
        t_staged = (town == g.tblk_u) & g.pg_ok_u
        pos = g.pos
        inrange = (pos >= torch.maximum(g.seq_minus_w, si * bs).view(B, 1, 1)) & (
            g.pos_lt_seq
        )
        t_valid = t_staged & inrange & (pos >= cut.view(B, 1, 1))

        if os.environ.get("VLLM_OSCAR_SINK_DIAG") and S_nb > 0:
            # Who lost their sink, and what is in the rows they should own?
            # `-1` means nothing was ever staged there (the request's prefill
            # never wrote, or the region was cleared); a different block id
            # means someone else's tag is sitting in this request's region.
            miss = (~sink_active) & (seq > S_eff)
            nmiss = int(miss.sum())
            k = getattr(self, "_sink_diag_n", 0)
            if nmiss and k < 12:
                self._sink_diag_n = k + 1
                r0 = int(torch.nonzero(miss).flatten()[0])
                g0 = int(layer._oscar_region_of[r0])
                rows = _rows_sink(g0, g.spg.view(-1), ppr)
                print(
                    f"[sink-diag] {getattr(layer, 'layer_name', '?')} "
                    f"{nmiss}/{B} without sink; first is slot {r0} "
                    f"region={g0} "
                    f"seq={int(seq[r0])} want_blk={bt[r0, :S_nb].tolist()} "
                    f"rows={rows.tolist()} "
                    f"owner_first_off={owner[rows, 0].tolist()} "
                    f"owner_last_off={owner[rows, bs - 1].tolist()}",
                    flush=True,
                )
        if os.environ.get("VLLM_OSCAR_STAGE_DEBUG") and getattr(
            self, "_dbg_n", 0
        ) < 3 and int(g.tblk_u.max()) > 0:
            # Fire only on a REAL batch.  The first version printed on the first
            # call and caught vLLM's dummy profiling run instead -- seq_lens all
            # 512, block_table all zeros, arena legitimately empty -- which looks
            # exactly like the bug being hunted.
            self._dbg_n = getattr(self, "_dbg_n", 0) + 1
            own_set = owner >= 0
            have = town
            print(
                f"[stage-dbg] B={B} rows={layer._oscar_stage_rows} ppr={ppr} "
                f"S_nb={S_nb} TP={TP} bs={bs}\n"
                f"[stage-dbg] owner rows holding any tag: "
                f"{int(own_set.any(dim=1).sum())}/{owner.shape[0]}   "
                f"tagged slots: {int(own_set.sum())}\n"
                f"[stage-dbg] tail rows the reader wants: "
                f"min={int(trow.min())} max={int(trow.max())}\n"
                f"[stage-dbg] of those slots, holding ANY tag: "
                f"{int((have >= 0).sum())}/{have.numel()}   "
                f"matching the wanted blk: "
                f"{int((have == g.tblk_u).sum())}\n"
                f"[stage-dbg] seq[:4]={seq[:4].tolist()} "
                f"cut[:4]={cut[:4].tolist()} si[:4]={si[:4].tolist()}\n"
                f"[stage-dbg] tblk[0,:3]={g.tblk_u[0, :3, 0].tolist()} "
                f"town[0,0,:3]={town[0, 0, :3].tolist()} "
                f"trow[0,:3]={trow[0, :3].tolist()}",
                flush=True,
            )

        seq_eff = (cut - si * bs).to(torch.int32)
        gidx = (g.gidx_base + si.unsqueeze(1)).clamp(max=maxpg - 1)
        bt_eff = torch.gather(bt, 1, gidx)
        q_rot = torch.matmul(query.float(), layer._oscar_Rk)
        lse1 = torch.full((B, Hq), float("-inf"), dtype=torch.float32, device=dev)
        out1 = OA.oscar_decode_attention(
            q_rot, kv_cache, bt_eff, seq_eff, self.scale,
            key_levels=self.cfg.key_levels,
            value_levels=self.cfg.value_levels,
            key_data_bytes=self.cfg.key_data_bytes,
            key_packed_size=self.cfg.key_packed_size,
            value_data_bytes=self.cfg.value_data_bytes,
            max_num_kv_splits=self.max_num_kv_splits,
            lse_buf=lse1,
        )
        o1 = torch.matmul(out1, layer._oscar_RvT)
        empty1 = (seq_eff <= 0).view(B, 1)
        lse1 = torch.where(
            empty1 | ~torch.isfinite(lse1),
            torch.full_like(lse1, float("-inf")), lse1,
        )
        o1 = torch.nan_to_num(o1)

        all_row = torch.cat([srow.expand(B, S_nb), trow], dim=1).contiguous()
        valid = torch.cat([s_valid, t_valid], dim=1).contiguous()
        o2, lse2 = self._bf16_tail(query, layer, all_row, valid)

        new_lse = torch.logaddexp(lse1, lse2)
        w1 = torch.exp(lse1 - new_lse).unsqueeze(-1)
        w2 = torch.exp(lse2 - new_lse).unsqueeze(-1)
        out = o1 * w1 + torch.nan_to_num(o2) * w2
        return out.to(query.dtype)

    Impl._ensure_staging = _ensure_staging
    Impl._region_map = _region_map_impl
    Impl._staging_write = _staging_write
    Impl._stage_splice = _stage_splice
    Impl._prefill_attention = _prefill_attention
    Impl._decode_attention_windowed = _decode_attention_windowed
    Impl._oscar_staging_is_per_request = True
    # Rebind the module-level name too: nothing else calls the stock bounds
    # kernel once _decode_attention_windowed is replaced, and going through the
    # module attribute is what lets the coverage probe wrap it.  Apply this
    # patch BEFORE oscar_staging_probe.apply().
    OA._oscar_window_bounds_kernel = _oscar_window_bounds_kernel_pr

    _PATCHED = True
    print("[oscar-staging-pr] per-request BF16 staging active "
          "(shared direct-mapped arena replaced)", flush=True)
    return True


def maybe_apply() -> bool:
    """Apply when VLLM_OSCAR_PER_REQUEST_STAGING=1. Off by default."""
    if os.environ.get("VLLM_OSCAR_PER_REQUEST_STAGING", "0") != "1":
        return False
    return apply_patch()
