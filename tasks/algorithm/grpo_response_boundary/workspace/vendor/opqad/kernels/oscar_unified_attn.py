"""Merge the BF16 sink/recent window into the INT2 split-K reduction.

Task #68.  Opt-in via ``QAT_OSCAR_UNIFIED_ATTN=1``; default OFF.

What this replaces
------------------
``_decode_attention_windowed`` currently runs **two** attentions per layer per
decoded token and merges them in eager torch::

    q_rot = query.float() @ R_k
    out1, lse1 = INT2 decode over the paged cache      (rotated basis)
    o1 = out1 @ R_v.T                                  (un-rotate, matmul #2)
    o2, lse2 = BF16 attention over the staging arena   (UNROTATED basis)
    new = logaddexp(lse1, lse2); out = o1*exp(lse1-new) + o2*exp(lse2-new)

Two bases force two attentions, and the softmax merge is ~6 eager ops on
``[B, Hq]`` / ``[B, Hq, D]`` per layer.  j104820 measured the window at **10.65
of 19.65 ms/step** at plen 8192 under full decode CUDA graphs -- 54% of decode,
and all of it device time now that capture has taken the dispatch half.

Upstream OSCAR (SGLang, ``decode_attention_fwd_int2_unified``) has nothing to
merge.  One shared split-K scratch, both tiers write slices of it, one
tier-agnostic stage-2 reduces all splits.  **The merge IS the split-K
reduction.**  ``triton_backend.py`` applies ``q @ R_k`` once and ``o @ R_v.T``
once, because the high-precision slots are stored *rotated too*: ``R_k``
orthogonal means ``(qR)·(kR) = q·k``, so one basis serves both tiers.

Note what upstream does NOT buy us: it still launches **two stage-1 kernels**
(the stock BF16 decode for HP slots, the INT2 kernel for quant slots).  Its
unified KV pool is about *addressing* -- one ``req_to_token`` telling it which
slot is which tier -- not about kernel count.  We keep our separate staging
arena and feed the same scratch from it, which gets the same reduction without
rewriting allocation, eviction, prefix caching and block tables.

Why a custom stage-2 is not optional
------------------------------------
The obvious port is "widen ``mid_o`` and keep calling the stock
``_fwd_kernel_stage2``".  That is silently wrong.  vLLM's stage-2
(``triton_decode_attention.py:580``) decides which splits to read by
*re-deriving* their ranges from the sequence length::

    kv_len_per_split = cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start   = kv_len_per_split * split_kv_id
    split_kv_end     = min(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    if split_kv_end > split_kv_start:   # only then accumulate

There is no per-split validity flag.  Widen the scratch and those ranges stop
corresponding to anything: for a short sequence only the first ``seq_len``
splits are non-empty, so window splits placed past that index are **dropped
without a word** -- the sink/recent protection would simply evaporate on short
sequences, which is the regime where it is cheapest to not notice.

Hence ``_oscar_unified_stage2``: loop every split, decide from the **LSE**, which
is the tier-agnostic signal.  That is also why upstream prefills ``attn_lse``
with ``-inf`` and why we must too -- ``_oscar_decode_stage1`` *returns early
without storing anything* on an empty split (``triton_oscar_decode.py:64``), so
an unwritten slot holds whatever was in the buffer last step, not ``-inf``.

Numerics
--------
NOT bitwise, unlike the CUDA-graph switch:

* the window moves into the rotated basis, so its scores become ``(qR)·(kR)``
  instead of ``q·k`` -- equal in exact arithmetic, ``max|RR^T - I| = 4.77e-07``
  in ours;
* the staging arena is BF16, so storing ``bf16(k @ R_k)`` adds one rounding;
* the reduction order changes: one online pass over ``1 + NUM_KV_SPLITS``
  splits instead of two independent softmaxes combined by ``logaddexp``.

Gate on ``max|d(out)|`` around 1e-5 and on ``val-topk/overlap_ratio``, never on
``out_sha``.

The trap this cost a day to find
--------------------------------
``_stage_splice`` is **not** a decode reader.  It splices staged BF16 rows over
the INT2-reconstructed prefix in ``_prefill_attention``, where ``k_cached`` and
``v_cached`` are in the **unrotated** basis.  Rotating the arena on write without
touching it feeds rotated K/V straight into prefill attention: wrong scores, no
error, and prefill is where the rollout's accuracy actually comes from.  So the
splice un-rotates on read.  It runs once per request per prefill, not per decode
step, so the cost is irrelevant.

The shared vLLM install at /shared/*/bitdistiller/envs/vllm-0.19.1-test/ is never
edited -- other people's jobs run out of it.  Everything here is an in-process
monkeypatch, applied after ``oscar_staging_patch``.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.triton_utils import tl, triton

# Geometry helpers are imported, not copied: this module forks only the ~25 lines
# of glue at the tail of `_decode_attention_windowed`, so a change to how rows,
# bounds or validity are computed is picked up here automatically.
from opqad.runtime.oscar_staging_patch import (
    _bounds,
    _cached,
    _decode_geom,
    _rows_sink,
    _rows_tail,
    _step_cache,
)

ENV_FLAG = "QAT_OSCAR_UNIFIED_ATTN"

# The window is 64 sink + 256 recent = 320 tokens against the INT2 tier's 8192+.
# One split per (batch, head) therefore does ~320 positions where an INT2 split
# does ~64 -- 5x the serial work, but there are 128x fewer such programs
# (B*Hq*1 vs B*Hq*NUM_KV_SPLITS), so it is nowhere near the critical path and
# splitting it further would only add scratch and reduction width.
HP_SPLITS = 1

_applied = False


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") not in ("", "0", "false", "False")


# ---------------------------------------------------------------------------
# Stage 1 for the high-precision tier.
#
# This is `_oscar_bf16_tail_tiled_kernel` (ours; verified against the eager path
# at max|d(o2)| = 3.8e-5 over 9192 real decode calls, job 33269, and already the
# eval default) with exactly one change: it writes its (normalised output, LSE)
# pair into a slice of the shared split-K scratch instead of into standalone
# `o2` / `lse2` tensors.  That needs one extra stride, because the LSE now lives
# at `[b, h, split, HEAD_DIM]` inside a 4-D buffer rather than at `[b, h]` in a
# 2-D one, so its head stride is no longer 1.
#
# The (normalised acc, m + log l) convention and the `-inf`-on-empty contract are
# already identical to `_oscar_decode_stage1`'s, which is what lets one stage-2
# read both tiers without knowing which is which.
# ---------------------------------------------------------------------------
@triton.jit
def _oscar_window_stage1_kernel(
    q_ptr, stage_k_ptr, stage_v_ptr, blk_ptr, valid_ptr,
    mid_ptr,
    stride_qb, stride_qh, stride_sr, stride_so, stride_sh,
    stride_bb, stride_vb, stride_vp,
    stride_mb, stride_mh, stride_ms,
    n_pages, block_sz, n_rows, n_kv_heads, gqa, head_dim, scale,
    BLOCK_D: tl.constexpr,
    BLOCK_BS: tl.constexpr,
):
    b = tl.program_id(0)
    kh = tl.program_id(1)
    gi = tl.program_id(2)

    d = tl.arange(0, BLOCK_D)
    dmask = d < head_dim
    o = tl.arange(0, BLOCK_BS)
    omask = o < block_sz

    qh_off = kh * gqa + gi
    q = tl.load(
        q_ptr + b * stride_qb + qh_off * stride_qh + d, mask=dmask, other=0.0
    ).to(tl.float32)

    # Finite sentinel, not -inf: the rescale below is unconditional and
    # `-inf - (-inf)` is NaN, which would poison `acc` on an all-invalid page.
    # It never reaches the output -- emptiness is decided by `l_i > 0`.
    m_i = -1.0e38
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    p = 0
    while p < n_pages:
        blk = tl.load(blk_ptr + b * stride_bb + p).to(tl.int64)
        row = blk % n_rows
        ok = tl.load(
            valid_ptr + b * stride_vb + p * stride_vp + o, mask=omask, other=0
        )
        base = (
            row * stride_sr + o[:, None] * stride_so + kh * stride_sh + d[None, :]
        )
        tile = omask[:, None] & dmask[None, :]
        k = tl.load(stage_k_ptr + base, mask=tile, other=0.0).to(tl.float32)
        v = tl.load(stage_v_ptr + base, mask=tile, other=0.0).to(tl.float32)

        keep = omask & (ok != 0)
        sc = tl.sum(q[None, :] * k, axis=1) * scale
        sc = tl.where(keep, sc, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(sc, axis=0))
        alpha = tl.exp(m_i - m_new)
        pexp = tl.where(keep, tl.exp(sc - m_new), 0.0)
        acc = acc * alpha + tl.sum(pexp[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(pexp, axis=0)
        m_i = m_new
        p += 1

    # Split 0 of the scratch.  Same layout as _oscar_decode_stage1: D floats of
    # normalised output, then the LSE at offset HEAD_DIM.
    out_base = b * stride_mb + qh_off * stride_mh  # + 0 * stride_ms
    out = tl.where(l_i > 0, acc / tl.maximum(l_i, 1e-38), 0.0)
    tl.store(mid_ptr + out_base + d, out, mask=dmask)
    lse = tl.where(l_i > 0, m_i + tl.log(tl.maximum(l_i, 1e-38)), float("-inf"))
    tl.store(mid_ptr + out_base + head_dim, lse)


# ---------------------------------------------------------------------------
# Tier-agnostic stage 2.
#
# vLLM's stock stage-2 re-derives each split's token range from `seq_lens` and
# skips the empty ones.  That is correct only when every split is a contiguous
# slice of one sequence, which stops being true the moment a second tier writes
# into the same buffer.  This one reads the LSE instead: `-inf` means "this split
# contributed nothing", which both tiers already emit.
# ---------------------------------------------------------------------------
@triton.jit
def _oscar_unified_stage2(
    mid_ptr, o_ptr, lse_ptr,
    stride_mb, stride_mh, stride_ms,
    stride_ob, stride_oh,
    stride_lb,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    d = tl.arange(0, BLOCK_D)
    dmask = d < HEAD_DIM

    base = b * stride_mb + h * stride_mh
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for s in range(0, NUM_SPLITS):
        off = base + s * stride_ms
        tlogic = tl.load(mid_ptr + off + HEAD_DIM)
        # An empty split contributes nothing.  Guarding on `tlogic > -inf`
        # rather than letting exp() do it keeps `e_max` finite, so the very
        # first non-empty split does not evaluate `exp(-inf - -inf)` = NaN.
        if tlogic > -float("inf"):
            tv = tl.load(mid_ptr + off + d, mask=dmask, other=0.0)
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc = acc * old_scale + tl.exp(tlogic - n_e_max) * tv
            e_sum = e_sum * old_scale + tl.exp(tlogic - n_e_max)
            e_max = n_e_max

    # Every split empty is legal (a request with nothing staged and an empty
    # INT2 range); emit zeros and -inf rather than dividing by zero.
    safe = tl.where(e_sum > 0.0, e_sum, 1.0)
    res = tl.where(e_sum > 0.0, acc / safe, 0.0)
    tl.store(o_ptr + b * stride_ob + h * stride_oh + d, res, mask=dmask)
    tl.store(
        lse_ptr + b * stride_lb + h,
        tl.where(e_sum > 0.0, e_max + tl.log(safe), -float("inf")),
    )


def _rot(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply a [D, D] (or per-head [H, D, D]) rotation to a [..., H, D] tensor."""
    if R.dim() == 2:
        return torch.matmul(x.float(), R)
    # Per-head: [N, H, D] x [H, D, D] -> [N, H, D]
    return torch.einsum("nhd,hde->nhe", x.float(), R)


def apply(verbose: bool = True) -> bool:
    """Install the unified-merge decode path.  Returns True if installed."""
    global _applied
    if _applied or not enabled():
        return False
    try:
        import vllm.v1.attention.backends.oscar_attn as OA
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[oscar-unified] not applied: {type(exc).__name__}: {exc}",
                  flush=True)
        return False

    Impl = OA.OscarAttentionImpl
    if not getattr(Impl, "_oscar_staging_is_per_request", False):
        # The per-request staging patch owns `_decode_attention_windowed` and
        # the arena layout this forks; without it the row/region arithmetic
        # below refers to a layout that is not there.
        if verbose:
            print("[oscar-unified] not applied: per-request staging is not "
                  "active (VLLM_OSCAR_PER_REQUEST_STAGING=1 is required)",
                  flush=True)
        return False

    _orig_staging_write = Impl._staging_write
    _orig_stage_splice = Impl._stage_splice

    def _staging_write_rotated(self, layer, key, value, attn_metadata):
        """Store the arena in the ROTATED basis so one query basis serves both
        tiers.  N is the number of tokens written this forward -- B (=16) in
        decode -- so this is a [16, Hk, D] x [D, D] matmul per layer, not a
        per-window-token cost."""
        Rk = getattr(layer, "_oscar_Rk", None)
        Rv = getattr(layer, "_oscar_Rv", None)
        if Rk is None or Rv is None:
            return _orig_staging_write(self, layer, key, value, attn_metadata)
        k_rot = _rot(key, Rk).to(key.dtype)
        v_rot = _rot(value, Rv).to(value.dtype)
        return _orig_staging_write(self, layer, k_rot, v_rot, attn_metadata)

    def _stage_splice_unrotated(self, layer, bt_row, cached_len, k_cached, v_cached):
        """Un-rotate what the splice pulls out of the arena.

        THE TRAP.  This is a PREFILL reader, not a decode one: it overwrites
        rows of the INT2-reconstructed prefix that `_prefill_attention` is about
        to attend over in the UNROTATED basis.  Feeding it rotated K/V is silent
        -- no shape error, no NaN, just wrong scores in the phase that produces
        the rollout's accuracy.  Runs once per request per prefill.
        """
        ks, vs = _orig_stage_splice(self, layer, bt_row, cached_len, k_cached, v_cached)
        Rk = getattr(layer, "_oscar_Rk", None)
        Rv = getattr(layer, "_oscar_Rv", None)
        if Rk is None or Rv is None or ks is k_cached:
            # `ks is k_cached` is the early-out path: nothing was spliced, so
            # there is nothing rotated to undo.
            return ks, vs
        kt = Rk.transpose(-1, -2) if Rk.dim() == 2 else Rk.transpose(-1, -2)
        vt = Rv.transpose(-1, -2) if Rv.dim() == 2 else Rv.transpose(-1, -2)
        return _rot(ks, kt).to(ks.dtype), _rot(vs, vt).to(vs.dtype)

    def _decode_attention_windowed_unified(self, query, kv_cache, attn_metadata, layer):
        """Fork of `oscar_staging_patch._decode_attention_windowed`.

        Everything down to `all_row` / `valid` is the same computation, reached
        through the same module-level helpers so it tracks changes there.  Only
        the tail differs: instead of a second attention plus a `logaddexp`, both
        tiers write into one scratch and one stage-2 reduces it.
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

        g = _decode_geom(attn_metadata, bt, seq, B, bs, S_nb, TP, S_eff, W,
                         maxpg, dev)
        offs = g.offs
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

        if S_nb > 0:
            srow = _rows_sink(rq, g.spg, ppr)
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

        seq_eff = (cut - si * bs).to(torch.int32)
        gidx = (g.gidx_base + si.unsqueeze(1)).clamp(max=maxpg - 1)
        bt_eff = torch.gather(bt, 1, gidx)

        all_row = torch.cat([srow.expand(B, S_nb), trow], dim=1).contiguous()
        valid = torch.cat([s_valid, t_valid], dim=1).contiguous()

        # ---- the shared split-K scratch -------------------------------------
        D = self.head_size
        n_int2 = self.max_num_kv_splits
        S_tot = HP_SPLITS + n_int2
        mid = getattr(layer, "_oscar_unified_mid", None)
        if (
            mid is None
            or mid.shape[0] < B
            or mid.shape[1] < Hq
            or mid.shape[2] != S_tot
        ):
            # Persistent per-layer buffer: a fresh allocation every decode step
            # would change addresses under CUDA-graph replay.
            mid = torch.empty(
                max(B, self.stage_slots), Hq, S_tot, D + 1,
                dtype=torch.float32, device=dev,
            )
            layer._oscar_unified_mid = mid
        mid_b = mid[:B]

        # `_oscar_decode_stage1` RETURNS EARLY WITHOUT STORING on an empty split
        # (triton_oscar_decode.py:64), so an unwritten LSE slot holds the
        # previous step's value.  The stock stage-2 never noticed because it
        # skipped exactly those splits by re-deriving the same ranges; ours
        # reads the LSE, so the buffer has to say -inf.  This is the same reason
        # upstream prefills `attn_lse`.
        mid_b[:, :, :, D].fill_(float("-inf"))

        # ---- tier 1: the BF16 window, into split 0 --------------------------
        P = all_row.shape[1]
        if P > 0:
            Hk = self.num_kv_heads
            gqa = Hq // Hk
            q_rot_w = torch.matmul(query.float(), layer._oscar_Rk).contiguous()
            vi = valid.to(torch.int8).contiguous()
            sk, sv = layer._oscar_stage_k, layer._oscar_stage_v
            _oscar_window_stage1_kernel[(B, Hk, gqa)](
                q_rot_w, sk, sv, all_row, vi, mid_b,
                q_rot_w.stride(0), q_rot_w.stride(1),
                sk.stride(0), sk.stride(1), sk.stride(2),
                all_row.stride(0), vi.stride(0), vi.stride(1),
                mid_b.stride(0), mid_b.stride(1), mid_b.stride(2),
                P, bs, layer._oscar_stage_rows, Hk, gqa, D, self.scale,
                BLOCK_D=triton.next_power_of_2(D),
                BLOCK_BS=triton.next_power_of_2(bs),
            )

        # ---- tier 2: INT2, into splits [HP_SPLITS:] -------------------------
        # The view keeps `mid_b`'s strides and shifts the base pointer, so the
        # INT2 kernel sees exactly the layout it always has and its own
        # NUM_KV_SPLITS is unchanged -- its per-split ranges, and therefore its
        # arithmetic, are identical to the two-attention path.
        q_rot = torch.matmul(query.float(), layer._oscar_Rk)
        OA.oscar_decode_attention(
            q_rot, kv_cache, bt_eff, seq_eff, self.scale,
            key_levels=self.cfg.key_levels,
            value_levels=self.cfg.value_levels,
            key_data_bytes=self.cfg.key_data_bytes,
            key_packed_size=self.cfg.key_packed_size,
            value_data_bytes=self.cfg.value_data_bytes,
            max_num_kv_splits=n_int2,
            mid_o_buf=mid_b[:, :, HP_SPLITS:, :],
        )

        # ---- one reduction, one un-rotation ---------------------------------
        out = torch.empty(B, Hq, D, dtype=torch.float32, device=dev)
        lse = torch.empty(B, Hq, dtype=torch.float32, device=dev)
        _oscar_unified_stage2[(B, Hq)](
            mid_b, out, lse,
            mid_b.stride(0), mid_b.stride(1), mid_b.stride(2),
            out.stride(0), out.stride(1),
            lse.stride(0),
            NUM_SPLITS=S_tot,
            BLOCK_D=triton.next_power_of_2(D),
            HEAD_DIM=D,
            num_warps=4,
            num_stages=2,
        )
        return torch.matmul(out, layer._oscar_RvT).to(query.dtype)

    Impl._staging_write = _staging_write_rotated
    Impl._stage_splice = _stage_splice_unrotated
    Impl._decode_attention_windowed = _decode_attention_windowed_unified
    _applied = True
    if verbose:
        print(
            "[oscar-unified] single-attention merge ON: staging arena stored "
            "rotated, both tiers reduced by one stage-2, logaddexp deleted. "
            "NOT bitwise vs the two-attention path -- out_sha will move; gate "
            "on max|d(out)| ~1e-5, not on the hash.",
            flush=True,
        )
    return True


def maybe_apply(verbose: bool = True) -> bool:
    return apply(verbose=verbose)
