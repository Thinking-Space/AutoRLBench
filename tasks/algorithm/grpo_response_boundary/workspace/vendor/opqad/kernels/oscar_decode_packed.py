"""INT2 decode stage 1 that loads each packed byte once instead of four times.

What this changes, and why
--------------------------
Both the stock ``_oscar_decode_stage1`` and our GQA variant unpack INT2 like
this::

    byte_idx = d_offs // 4                 # [BLOCK_D], every value repeated 4x
    bit_shift = (d_offs % 4) * 2
    k_byte = tl.load(base[:, None] + byte_idx[None, :], ...)   # [BLOCK_KV, BLOCK_D]
    q_k = ((k_byte >> bit_shift[None, :]) & 3).to(tl.float32)

``byte_idx`` is ``0,0,0,0,1,1,1,1,...``: the tile issues ``BLOCK_D`` load lanes
to fetch ``BLOCK_D/4`` distinct bytes, and then holds the 4x-inflated result in
registers as int32 before widening it again to fp32. At HEAD_DIM 128 that is a
``[BLOCK_KV, 128]`` int32 tile where a ``[BLOCK_KV, 32]`` one would do.

The same loop reads the per-(token, head) scale and zero as **eight separate
scalar byte loads** -- four for K, four for V -- to reconstruct two fp16 values
out of four contiguous bytes. Each is a masked gather whose lanes are
``stride_cache_pos`` apart, so none of them coalesce.

This module does both loads at their true width:

* the packed tile is loaded as ``[BLOCK_KV, HEAD_DIM//4]`` and widened by a
  shift along a new axis, then reshaped. The ordering is exact, not
  approximate: element ``(kv, b, s)`` of the ``[BLOCK_KV, HEAD_DIM//4, 4]``
  intermediate lands at ``d = b*4 + s`` after a row-major reshape, and the
  stock kernel's own indexing is ``b = d//4``, ``s = d%4``. Same permutation.
* the four metadata bytes are one ``[BLOCK_KV, 4]`` load, and the two fp16
  values are recovered with a masked shift-and-sum. The two byte contributions
  occupy disjoint bit ranges, so the sum is an exact ``lo | (hi << 8)``.

Why this is the right lever
---------------------------
The kernel measures 105-145 GB/s against a ~864 GB/s roofline (12-17%), and a
full sweep of NUM_KV_SPLITS / BLOCK_KV / num_warps / num_stages moves it 1.01x
at the large shapes. It is not bandwidth-bound. Both the stock tuner and the
GQA kernel converge on ``BLOCK_KV=2-4, num_warps=1``, which is the signature of
register pressure -- and the 4x-inflated packed tile is where that pressure
comes from.

This composes with GQA reuse rather than replacing it: GQA removes traffic that
is redundant *across query heads*, this removes lanes that are redundant
*within one vector*. The kernel below keeps the GQA regrid.

Numerics
--------
Bit-exact with the GQA kernel by construction -- same values, same order of
accumulation, only the loads differ. ``bench_packed_decode.py`` asserts exact
equality against both the stock and the GQA launcher before it times anything;
a mismatch means the reshape permutation or the metadata reassembly is wrong,
and either would be silent otherwise.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from opqad.kernels.oscar_decode_gqa import _fwd_kernel_stage2


@triton.jit
def _oscar_decode_stage1_packed(
    Q_rot_ptr,        # [B, Hq, D] fp32, already rotated by R_k
    KV_cache_ptr,     # [num_blocks, block_size, Hk, slot_size] uint8
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,     # [B] int32
    Mid_o_ptr,        # [B, Hq, NUM_KV_SPLITS, D+1] fp32
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,
    KEY_PACKED: tl.constexpr,
    VALUE_DATA_BYTES: tl.constexpr,
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_B: tl.constexpr,   # BLOCK_D // 4, the packed tile's true width
    BLOCK_KV: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    bid = tl.program_id(0)
    kv_head = tl.program_id(1)
    sid = tl.program_id(2)

    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)
    g_offs = tl.arange(0, BLOCK_G)
    g_mask = g_offs < KV_GROUP_SIZE

    # The packed tile at its true width, plus the within-byte shifts.  Four
    # 2-bit values live in each byte, least significant first, which is the
    # stock kernel's `(d % 4) * 2`.
    b_offs = tl.arange(0, BLOCK_B)
    b_mask = b_offs < (HEAD_DIM // 4)
    s_offs = tl.arange(0, 4)
    shifts = s_offs * 2
    # Metadata: bytes 0,1 are the fp16 scale and 2,3 the fp16 zero, little
    # endian.  Both shift vectors are clamped to {0, 8} on the lanes they do not
    # own: `tl.where` discards those lanes afterwards, but the shift is still
    # *evaluated* on them, and an unclamped `s*8` would put byte 3 at `<< 24`,
    # which overflows the int32 sign bit.  Cheaper to not generate the value.
    shsc = tl.where(s_offs < 2, s_offs * 8, 0)
    shz = tl.where(s_offs >= 2, (s_offs - 2) * 8, 0)
    lo2 = s_offs < 2
    hi2 = s_offs >= 2

    q_head0 = kv_head * KV_GROUP_SIZE
    q_ptrs = (
        Q_rot_ptr
        + bid * stride_qb
        + (q_head0 + g_offs)[:, None] * stride_qh
        + d_offs[None, :]
    )
    q_rot = tl.load(
        q_ptrs, mask=g_mask[:, None] & d_mask[None, :], other=0.0
    ).to(tl.float32)                                        # [BLOCK_G, BLOCK_D]

    m_prev = tl.zeros([BLOCK_G], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_G], dtype=tl.float32)
    acc = tl.zeros([BLOCK_G, BLOCK_D], dtype=tl.float32)
    bt_base = bid * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx, mask=kv_mask, other=0
        ).to(tl.int64)
        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # ---- K: one byte per byte, then widen ----
        k_byte = tl.load(
            KV_cache_ptr + slot_bases[:, None] + b_offs[None, :],
            mask=kv_mask[:, None] & b_mask[None, :],
            other=0,
        ).to(tl.int32)                                      # [BLOCK_KV, BLOCK_B]
        q_k = tl.reshape(
            (k_byte[:, :, None] >> shifts[None, None, :]) & (KEY_LEVELS - 1),
            (BLOCK_KV, BLOCK_D),
        ).to(tl.float32)

        k_meta = slot_bases + KEY_DATA_BYTES
        km = tl.load(
            KV_cache_ptr + k_meta[:, None] + s_offs[None, :],
            mask=kv_mask[:, None],
            other=0,
        ).to(tl.int32)                                      # [BLOCK_KV, 4]
        k_scale = (
            tl.sum(tl.where(lo2[None, :], km << shsc[None, :], 0), axis=1)
            .to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        )
        k_zero = (
            tl.sum(tl.where(hi2[None, :], km << shz[None, :], 0), axis=1)
            .to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        )

        k_deq = q_k * k_scale[:, None] + k_zero[:, None]     # [BLOCK_KV, BLOCK_D]

        scores = tl.sum(
            tl.where(
                d_mask[None, None, :],
                q_rot[:, None, :] * k_deq[None, :, :],
                0.0,
            ),
            axis=2,
        ) * ATTN_SCALE
        scores = tl.where(kv_mask[None, :], scores, -float("inf"))

        n_e_max = tl.maximum(tl.max(scores, 1), m_prev)      # [BLOCK_G]
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max[:, None])                # [BLOCK_G, BLOCK_KV]

        # ---- V: same treatment ----
        v_base = slot_bases + KEY_PACKED
        v_byte = tl.load(
            KV_cache_ptr + v_base[:, None] + b_offs[None, :],
            mask=kv_mask[:, None] & b_mask[None, :],
            other=0,
        ).to(tl.int32)
        q_v = tl.reshape(
            (v_byte[:, :, None] >> shifts[None, None, :]) & (VALUE_LEVELS - 1),
            (BLOCK_KV, BLOCK_D),
        ).to(tl.float32)

        v_meta = v_base + VALUE_DATA_BYTES
        vm = tl.load(
            KV_cache_ptr + v_meta[:, None] + s_offs[None, :],
            mask=kv_mask[:, None],
            other=0,
        ).to(tl.int32)
        v_scale = (
            tl.sum(tl.where(lo2[None, :], vm << shsc[None, :], 0), axis=1)
            .to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        )
        v_zero = (
            tl.sum(tl.where(hi2[None, :], vm << shz[None, :], 0), axis=1)
            .to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        )
        values = q_v * v_scale[:, None] + v_zero[:, None]    # [BLOCK_KV, BLOCK_D]

        acc = acc * re_scale[:, None] + tl.sum(
            p[:, :, None] * values[None, :, :], axis=1
        )
        l_prev = l_prev * re_scale + tl.sum(p, 1)
        m_prev = n_e_max

    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    out_base = (
        bid * stride_mid_b
        + (q_head0 + g_offs) * stride_mid_h
        + sid * stride_mid_s
    )
    tl.store(
        Mid_o_ptr + out_base[:, None] + d_offs[None, :],
        acc / safe_l[:, None],
        mask=g_mask[:, None] & d_mask[None, :],
    )
    tl.store(
        Mid_o_ptr + out_base + HEAD_DIM,
        m_prev + tl.log(safe_l),
        mask=g_mask,
    )


def oscar_decode_attention_packed(
    q_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    key_levels: int,
    value_levels: int,
    key_data_bytes: int,
    key_packed_size: int,
    value_data_bytes: int,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    max_num_kv_splits: int = 16,
    block_kv: int = 4,
    num_warps: int = 1,
    num_stages: int = 1,
) -> torch.Tensor:
    """Drop-in replacement for ``oscar_decode_attention``; same signature as the
    GQA launcher, so the three can be swapped in a bench without special-casing.

    Stage 2 is the stock ``_fwd_kernel_stage2``: the Mid_o split layout is
    unchanged, only who fills it changes.
    """
    B, Hq, D = q_rot.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = q_rot.device
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_G = triton.next_power_of_2(kv_group_size)
    NUM_KV_SPLITS = max_num_kv_splits

    # The reshape needs BLOCK_B * 4 == BLOCK_D exactly, and the packed layout
    # needs HEAD_DIM divisible by 4.  Both hold for every OSCAR head_dim we
    # serve (128), but a wrong BLOCK_D would corrupt silently, so assert.
    if D % 4 != 0 or BLOCK_D % 4 != 0:
        raise ValueError(
            f"packed decode needs head_dim and its power-of-2 pad divisible by "
            f"4; got head_dim={D}, BLOCK_D={BLOCK_D}"
        )
    BLOCK_B = BLOCK_D // 4

    q_rot = q_rot.contiguous().float()

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=device
        )

    _oscar_decode_stage1_packed[(B, Hk, NUM_KV_SPLITS)](
        q_rot, kv_cache, block_table, seq_lens, mid_o,
        q_rot.stride(0), q_rot.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size,
        KEY_DATA_BYTES=key_data_bytes,
        KEY_PACKED=key_packed_size,
        VALUE_DATA_BYTES=value_data_bytes,
        KEY_LEVELS=key_levels,
        VALUE_LEVELS=value_levels,
        ATTN_SCALE=scale,
        BLOCK_D=BLOCK_D,
        BLOCK_B=BLOCK_B,
        BLOCK_KV=block_kv,
        BLOCK_G=BLOCK_G,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if output_buf is not None and output_buf.shape[0] >= B:
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)

    _fwd_kernel_stage2[(B, Hq)](
        mid_o, output, lse, seq_lens,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        output.stride(0), output.stride(1), lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS, BLOCK_DV=BLOCK_D, Lv=D, OUTPUT_FP16=0,
        num_warps=4, num_stages=2,
    )
    return output
