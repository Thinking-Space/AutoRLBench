"""Optimized fused RoPE for THD (packed variable-length sequences).

Replaces TE's fused_apply_rotary_pos_emb_thd which launches
grid=(max_s, n_seqs) CUDA blocks.  With many spans, >99% of blocks
just early-exit, wasting GPU scheduling bandwidth.

This kernel uses grid=(total_tokens,) with O(log n_seqs) binary
search per token -- cost is constant regardless of span count.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
@triton.jit
def _rope_thd_fwd(
    T, O, F, CS,
    n_seqs,
    stride_t, stride_h, stride_d,
    cp_rank,
    D2: tl.constexpr,
    H: tl.constexpr,
    HALF: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    TAIL: tl.constexpr,
):
    pid = tl.program_id(0)

    # ---- binary search: find span containing this token ----
    lo = tl.zeros([], tl.int32)
    hi = n_seqs + lo
    for _ in range(20):
        mid = (lo + hi) >> 1
        v = tl.load(CS + mid + 1).to(tl.int32)
        c = (v // CP <= pid) if CP > 1 else (v <= pid)
        lo = tl.where(c, mid + 1, lo)
        hi = tl.where(c, hi, mid)

    # ---- compute freq position (local pos within span) ----
    raw_start = tl.load(CS + lo).to(tl.int32)
    start = (raw_start // CP) if CP > 1 else raw_start
    lp = pid - start

    if CP > 1:
        raw_end = tl.load(CS + lo + 1).to(tl.int32)
        cl = raw_end // CP - start
        hl = cl >> 1
        fp = tl.where(
            lp < hl,
            lp + cp_rank * hl,
            cl * CP - (cp_rank + 1) * hl + lp - hl,
        )
    else:
        fp = lp

    # ---- load freqs & compute cos/sin for all d2 dims ----
    d_id = tl.arange(0, D2)
    f = tl.load(F + fp * D2 + d_id)
    cos = tl.cos(f)
    sin = tl.sin(f)

    is_first = d_id < HALF
    rotate_idx = tl.where(is_first, d_id + HALF, d_id - HALF)

    # ---- apply rotation per head (unified expression matching TE) ----
    for h in range(H):
        base = pid * stride_t + h * stride_h
        v_src = tl.load(T + base + d_id * stride_d).to(tl.float32)
        v_src_rotate = tl.load(T + base + rotate_idx * stride_d).to(tl.float32)
        v_src_rotate = tl.where(is_first, -v_src_rotate, v_src_rotate)
        out = v_src * cos + v_src_rotate * sin
        tl.store(O + base + d_id * stride_d, out.to(T.dtype.element_ty))

        if TAIL > 0:
            tail_ix = tl.arange(0, TAIL) + D2
            tail_mask = tail_ix < D
            tl.store(
                O + base + tail_ix * stride_d,
                tl.load(T + base + tail_ix * stride_d, mask=tail_mask),
                mask=tail_mask,
            )


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------
@triton.jit
def _rope_thd_bwd(
    GO, GI, F, CS,
    n_seqs,
    stride_t, stride_h, stride_d,
    cp_rank,
    D2: tl.constexpr,
    H: tl.constexpr,
    HALF: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    TAIL: tl.constexpr,
):
    pid = tl.program_id(0)

    lo = tl.zeros([], tl.int32)
    hi = n_seqs + lo
    for _ in range(20):
        mid = (lo + hi) >> 1
        v = tl.load(CS + mid + 1).to(tl.int32)
        c = (v // CP <= pid) if CP > 1 else (v <= pid)
        lo = tl.where(c, mid + 1, lo)
        hi = tl.where(c, hi, mid)

    raw_start = tl.load(CS + lo).to(tl.int32)
    start = (raw_start // CP) if CP > 1 else raw_start
    lp = pid - start

    if CP > 1:
        raw_end = tl.load(CS + lo + 1).to(tl.int32)
        cl = raw_end // CP - start
        hl = cl >> 1
        fp = tl.where(
            lp < hl,
            lp + cp_rank * hl,
            cl * CP - (cp_rank + 1) * hl + lp - hl,
        )
    else:
        fp = lp

    d_id = tl.arange(0, D2)
    f = tl.load(F + fp * D2 + d_id)
    cos = tl.cos(f)
    sin = tl.sin(f)

    # Backward rotation: for d_id < half, sin uses freq[d_id + half];
    # for d_id >= half, sin uses -freq[d_id - half].
    # This matches TE's backward kernel exactly.
    is_first = d_id < HALF
    rotate_idx = tl.where(is_first, d_id + HALF, d_id - HALF)
    cross_sin = tl.sin(tl.load(F + fp * D2 + tl.where(is_first, d_id + HALF, d_id + HALF - D2)))
    cross_sin = tl.where(is_first, cross_sin, -cross_sin)

    for h in range(H):
        base = pid * stride_t + h * stride_h
        g_src = tl.load(GO + base + d_id * stride_d).to(tl.float32)
        g_rotate = tl.load(GO + base + rotate_idx * stride_d).to(tl.float32)
        out = g_src * cos + g_rotate * cross_sin
        tl.store(GI + base + d_id * stride_d, out.to(GO.dtype.element_ty))

        if TAIL > 0:
            tail_ix = tl.arange(0, TAIL) + D2
            tail_mask = tail_ix < D
            tl.store(
                GI + base + tail_ix * stride_d,
                tl.load(GO + base + tail_ix * stride_d, mask=tail_mask),
                mask=tail_mask,
            )


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------
class _FusedRoPETHD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, cu_seqlens, freqs, cp_size, cp_rank):
        t = t.contiguous()
        total_tokens, h, d = t.shape
        d2 = freqs.size(-1)
        half = d2 // 2
        n_seqs = cu_seqlens.numel() - 1
        tail = d - d2

        out = torch.empty_like(t)
        freqs_flat = freqs.contiguous().view(-1)

        _rope_thd_fwd[(total_tokens,)](
            t, out, freqs_flat, cu_seqlens,
            n_seqs,
            t.stride(0), t.stride(1), t.stride(2),
            cp_rank,
            D2=d2, H=h, HALF=half, CP=cp_size,
            D=d, TAIL=triton.next_power_of_2(tail) if tail > 0 else 0,
        )

        ctx.save_for_backward(cu_seqlens, freqs)
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        return out

    @staticmethod
    def backward(ctx, grad_output):
        cu_seqlens, freqs = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        total_tokens, h, d = grad_output.shape
        d2 = freqs.size(-1)
        half = d2 // 2
        n_seqs = cu_seqlens.numel() - 1
        tail = d - d2

        grad_input = torch.empty_like(grad_output)
        freqs_flat = freqs.contiguous().view(-1)

        _rope_thd_bwd[(total_tokens,)](
            grad_output, grad_input, freqs_flat, cu_seqlens,
            n_seqs,
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            ctx.cp_rank,
            D2=d2, H=h, HALF=half, CP=ctx.cp_size,
            D=d, TAIL=triton.next_power_of_2(tail) if tail > 0 else 0,
        )

        return grad_input, None, None, None, None


def fused_rope_thd(t, cu_seqlens, freqs, cp_size=1, cp_rank=0):
    """Drop-in replacement for TE's fused_apply_rotary_pos_emb_thd.

    Grid = (total_tokens,) — O(1) w.r.t. span count.
    """
    return _FusedRoPETHD.apply(t, cu_seqlens, freqs, cp_size, cp_rank)
