"""INT2 decode attention with GQA reuse: dequantize each KV page once, not once per query head.

The measurement this implements
-------------------------------
P1b (job 51275) put the INT2 dequant/score kernel at 52-56% of OSCAR decode time
at every realistic batch and context, and the only term growing with both.
Launch-parameter tuning is exhausted (job 52158: 1.01-1.10x over the hardcoded
BLOCK_KV=4 / num_warps=1 / num_stages=1).

Job 52268 found where the time goes.  Holding Hk, B and ctx fixed so the unique
bytes are identical and varying only Hq:

    group  GB/s on unique bytes   GB/s x group
      1            558                 558
      2            286                 571
      4            145                 581
      8             73                 586

7.61x spread on unique bytes; 1.05x on GB/s-x-group.  The kernel is saturated at
~570 GB/s -- 66% of the L20's ~864 -- on traffic that is `kv_group_size`
redundant.  The stock grid is (B, Hq, splits) with `kv_head = hid //
KV_GROUP_SIZE`, so all G query heads sharing a KV head each load and dequantize
the same pages; blocks dispatch x-fastest, so those siblings are B blocks apart
and do not find each other in L2.

The change
----------
Regrid to (B, Hk, splits).  One program owns a KV head and all G of its query
heads: load the K/V tile once, unpack it once, score G queries against it, and
carry G independent online-softmax states.

    stock:   acc [BLOCK_D]        m, l scalars      1 query  per tile load
    here:    acc [BLOCK_G, BLOCK_D]  m, l [BLOCK_G]    G queries per tile load

G is 4-8 in the models here, below tl.dot's 16-minimum, so the score and the
value accumulation are broadcast reductions rather than MMA.  That trades
arithmetic for bandwidth, which is the right direction for a kernel measured at
66% of roofline on redundant reads.

BLOCK_G is the next power of two above G and the surplus rows are masked; G=6
(the 1.5B: 12 query heads over 2 KV heads) is not a power of two.

Correctness
-----------
Not wired into anything by default.  bench_oscar_decode_kernel.py checks every
configuration against the stock launcher's output before reporting its time, and
nothing here is used in an eval until that passes -- a faster kernel that is
wrong is not a result.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2


@triton.jit
def _oscar_decode_stage1_gqa(
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
    BLOCK_KV: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    bid = tl.program_id(0)
    kv_head = tl.program_id(1)   # <- was the QUERY head in the stock kernel
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

    # INT2 unpack index vectors (loop-invariant): 4 indices per byte.
    byte_idx = d_offs // 4
    bit_shift = (d_offs % 4) * 2

    # The G query heads that share this KV head are contiguous in Hq, because the
    # stock kernel's mapping is kv_head = hid // KV_GROUP_SIZE.
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

        # ---- dequant K once for all G query heads ----
        k_byte = tl.load(
            KV_cache_ptr + slot_bases[:, None] + byte_idx[None, :],
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        q_k = ((k_byte >> bit_shift[None, :]) & (KEY_LEVELS - 1)).to(tl.float32)

        k_meta = slot_bases + KEY_DATA_BYTES
        ksc_lo = tl.load(KV_cache_ptr + k_meta, mask=kv_mask, other=0).to(tl.uint16)
        ksc_hi = tl.load(KV_cache_ptr + k_meta + 1, mask=kv_mask, other=0).to(tl.uint16)
        k_scale = (ksc_lo | (ksc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        kzr_lo = tl.load(KV_cache_ptr + k_meta + 2, mask=kv_mask, other=0).to(tl.uint16)
        kzr_hi = tl.load(KV_cache_ptr + k_meta + 3, mask=kv_mask, other=0).to(tl.uint16)
        k_zero = (kzr_lo | (kzr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        k_deq = q_k * k_scale[:, None] + k_zero[:, None]     # [BLOCK_KV, BLOCK_D]

        # [BLOCK_G, BLOCK_KV]: G queries against the one dequantized tile.
        # tl.dot needs >=16 in every dim and G is 4-8, so this stays a broadcast
        # reduction.
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

        # ---- dequant V once for all G query heads ----
        v_base = slot_bases + KEY_PACKED
        v_byte = tl.load(
            KV_cache_ptr + v_base[:, None] + byte_idx[None, :],
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        q_v = ((v_byte >> bit_shift[None, :]) & (VALUE_LEVELS - 1)).to(tl.float32)

        v_meta = v_base + VALUE_DATA_BYTES
        vsc_lo = tl.load(KV_cache_ptr + v_meta, mask=kv_mask, other=0).to(tl.uint16)
        vsc_hi = tl.load(KV_cache_ptr + v_meta + 1, mask=kv_mask, other=0).to(tl.uint16)
        v_scale = (vsc_lo | (vsc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        vzr_lo = tl.load(KV_cache_ptr + v_meta + 2, mask=kv_mask, other=0).to(tl.uint16)
        vzr_hi = tl.load(KV_cache_ptr + v_meta + 3, mask=kv_mask, other=0).to(tl.uint16)
        v_zero = (vzr_lo | (vzr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
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


def oscar_decode_attention_gqa(
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
    """Drop-in replacement for oscar_decode_attention with GQA reuse.

    Stage 2 (the log-sum-exp reduction across splits) is untouched and still the
    stock `_fwd_kernel_stage2`: the split layout in Mid_o is identical, only who
    fills it changes.
    """
    B, Hq, D = q_rot.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = q_rot.device
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_G = triton.next_power_of_2(kv_group_size)
    NUM_KV_SPLITS = max_num_kv_splits

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

    _oscar_decode_stage1_gqa[(B, Hk, NUM_KV_SPLITS)](
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
