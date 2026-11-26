"""Opt-in: swap the TurboQuant k2v3 decode kernel for a ported "v2" kernel.

`tq_decode_tune.py` (QAT_TQ_DECODE_TUNE) exhausted the launch-config space
(BLOCK_KV/warps/stages) for the STOCK kernel and found it flat -- k2v3 decode
stays ~102.6ms/step vs BF16's 9.18ms (11.2x). Direct source read of the
installed kernel (`vllm/v1/attention/ops/triton_turboquant_decode.py`,
`_tq_decode_stage1`) shows the bottleneck is algorithmic, not a tuning miss:

  - `scores = tl.sum(Q[None,:] * K, axis=1)` -- scalar reduction, no
    `tl.dot`, no tensor cores.
  - grid = (B, Hq, NUM_KV_SPLITS) -- one program PER QUERY HEAD. With
    KV_GROUP_SIZE = Hq/Hk = 4 (4B model), the same KV segment is
    independently reloaded/dequantized 4x per split.

Upstream vLLM has a real fix for this, but it is not a ready artifact:
`vllm/v1/attention/ops/turboquant_soa/triton_turboquant_decode_v2.py` on
main is a 97-line stub (helper functions only, no kernel). The actual
675-line kernel lives only in still-Draft PR vllm-project/vllm#40396
(author aditi-amd), tested only on MI300X/ROCm -- no NVIDIA numbers exist
anywhere. This module is a PORT of that draft kernel, adapted to our call
site and our CUDA hardware, not an import (the draft PR is an unmerged
fork branch and isn't installed here).

What changed vs. the draft PR
------------------------------
One correctness fix, everything else ported as designed:

  - `_get_pair_lut` here does NOT cache by `centroids.data_ptr()`. The
    draft PR's version does, but vLLM main's own (stub-only) revision of
    the same file explicitly dropped that cache with a comment: CUDA
    allocator memory reuse across different centroid tensors can silently
    return a stale LUT. Rebuilding on every call costs a few KB touch (
    N*N*2*4 bytes, 2KB for 4-bit MSE) against ~30720-ctx decode attention;
    not worth reintroducing a bug upstream itself already moved past.

  - Our real call site (`vllm.v1.attention.backends.turboquant_attn`,
    confirmed via `tq_decode_tune.py`'s own copy of the signature) never
    passes `value_packed_size` or `max_seq_len`; the draft PR's launcher
    declares both but never reads them (`del max_seq_len`;
    `value_packed_size` isn't referenced anywhere in the kernel body
    either). Dropped from this port's signature entirely rather than kept
    as unused required params.

Numerics
--------
This is NOT a launch-config no-op like tq_decode_tune.py. `tl.dot` with
fp16-cast Q/K/P/V inputs computes the QK score and the P.V accumulation on
tensor cores in fp16, vs. the stock kernel's fp32 scalar reduction. This
CAN perturb logits. Any use of this patch must be verified via a greedy
token-diff + paired |logprob-delta| pass (see
experiments/2026-09-22/tq_decode_tune/verify_v2.py) before being trusted
for anything beyond a throughput measurement.

Default off
-----------
QAT_TQ_DECODE_V2=1. Mutually exclusive with QAT_TQ_DECODE_TUNE (checked in
opqad/integrations/vllm_plugin.py): v2 replaces the whole launcher, v1 only retunes its
launch params, and applying both would mean the second one silently wins.
"""
from __future__ import annotations

import os
from typing import Any

import torch

ENV_FLAG = "QAT_TQ_DECODE_V2"

_applied = False


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") not in ("", "0", "false", "False")


def _build_pair_lut(centroids: torch.Tensor) -> torch.Tensor:
    """[N,N,2] fp32 pair LUT: pair_lut[i,j] = (centroids[i], centroids[j]),
    flattened to [N*N, 2]. Rebuilt on every call -- see module docstring for
    why this does not cache by data_ptr()."""
    N = centroids.shape[0]
    c = centroids.float()
    lut = torch.empty(N, N, 2, dtype=torch.float32, device=centroids.device)
    lut[:, :, 0] = c[:, None]
    lut[:, :, 1] = c[None, :]
    return lut.reshape(N * N, 2).contiguous()


def apply(verbose: bool = True) -> bool:
    """Patch the in-process k2v3 decode kernel with the ported v2 launcher.

    Returns True if applied. Safe to call when vLLM has no TurboQuant
    backend loaded (an OSCAR or BF16 arm): the import fails and this
    returns False rather than raising. Must be called before the first
    decode call, i.e. before LLM(...) or, for training, before rollout
    starts.
    """
    global _applied
    if _applied or not enabled():
        return False
    try:
        from vllm.triton_utils import tl, triton
        from vllm.v1.attention.ops.triton_decode_attention import (
            _fwd_kernel_stage2,
        )
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            _get_layout,
            _use_fp8_e4b15,
        )
        import vllm.v1.attention.backends.turboquant_attn as _tqa
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[tq-v2] not applied: {type(exc).__name__}: {exc}",
                  flush=True)
        return False

    @triton.jit
    def _tq_decode_stage1_v2(
        Q_rot_ptr,
        KV_cache_ptr,
        Block_table_ptr,
        Seq_lens_ptr,
        Centroids_ptr,
        Pair_lut_ptr,
        Mid_o_ptr,
        stride_qb,
        stride_qh,
        stride_cache_block,
        stride_cache_pos,
        stride_cache_head,
        stride_bt_b,
        stride_mid_b,
        stride_mid_h,
        stride_mid_s,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        PADDED_SLOT: tl.constexpr,
        MAX_NUM_BLOCKS: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr,
        KV_GROUP_SIZE: tl.constexpr,
        MSE_BITS: tl.constexpr,
        MSE_BYTES: tl.constexpr,
        KPS: tl.constexpr,
        VQB: tl.constexpr,
        VAL_DATA_BYTES: tl.constexpr,
        N_CENTROIDS: tl.constexpr,
        ATTN_SCALE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        KEY_FP8: tl.constexpr,
        NORM_CORRECTION: tl.constexpr = 0,
        FP8_E4B15: tl.constexpr = 0,
        USE_PAIR_LUT: tl.constexpr = 0,
    ):
        # Grid: (B, Hk, NUM_KV_SPLITS) -- one program per KV head group,
        # covering all KV_GROUP_SIZE query heads sharing that KV head.
        bid = tl.program_id(0)
        kv_hid = tl.program_id(1)
        sid = tl.program_id(2)

        seq_len = tl.load(Seq_lens_ptr + bid)
        split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
        split_start = split_len * sid
        split_end = tl.minimum(split_start + split_len, seq_len)

        if split_start >= split_end:
            return

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < HEAD_DIM
        m_offs = tl.arange(0, BLOCK_M)
        t_offs = tl.arange(0, TILE_SIZE)

        q_head_offs = kv_hid * KV_GROUP_SIZE + m_offs
        q_mask = m_offs < KV_GROUP_SIZE

        q_addrs = bid * stride_qb + q_head_offs[:, None] * stride_qh + d_offs[None, :]
        Q = tl.load(
            Q_rot_ptr + q_addrs,
            mask=q_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        if not KEY_FP8:
            _c_warm = tl.load(Centroids_ptr + tl.arange(0, N_CENTROIDS))  # noqa: F841

        if not KEY_FP8:
            if MSE_BITS == 4:
                pass
            else:
                mse_bit_off = d_offs * MSE_BITS
                mse_byte_idx = mse_bit_off // 8
                mse_bit_shift = mse_bit_off % 8
                mse_mask_val = (1 << MSE_BITS) - 1

        if VQB == 3:
            val_bit_off = d_offs * 3
            val_byte_idx = val_bit_off // 8
            val_bit_shift = val_bit_off % 8

        RCP_LN2: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471805599453
        QK_SCALE = ATTN_SCALE * RCP_LN2

        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        L = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        bt_base = bid * stride_bt_b

        for start_n in range(split_start, split_end, TILE_SIZE):
            kv_offs = start_n + t_offs
            kv_mask_1d = kv_offs < split_end

            page_idx = kv_offs // BLOCK_SIZE
            page_off = kv_offs % BLOCK_SIZE
            block_nums = tl.load(
                Block_table_ptr + bt_base + page_idx,
                mask=kv_mask_1d,
                other=0,
            ).to(tl.int64)

            slot_bases = (
                block_nums * stride_cache_block
                + page_off.to(tl.int64) * stride_cache_pos
                + tl.cast(kv_hid, tl.int64) * stride_cache_head
            )

            if KEY_FP8:
                k_addrs = slot_bases[:, None] + d_offs[None, :]
                k_raw = tl.load(
                    KV_cache_ptr + k_addrs,
                    mask=kv_mask_1d[:, None] & d_mask[None, :],
                    other=0,
                )
                if FP8_E4B15:
                    K_f = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
                else:
                    K_f = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                K_T = tl.trans(K_f)
                S = QK_SCALE * tl.dot(Q.to(tl.float16), K_T.to(tl.float16))
            else:
                if MSE_BITS == 4 and USE_PAIR_LUT:
                    HALF_D: tl.constexpr = BLOCK_D // 2
                    half_offs = tl.arange(0, HALF_D)
                    byte_mask = (half_offs * 2) < HEAD_DIM

                    byte_addrs = slot_bases[:, None] + half_offs[None, :]
                    byte_raw = tl.load(
                        KV_cache_ptr + byte_addrs,
                        mask=kv_mask_1d[:, None] & byte_mask[None, :],
                        other=0,
                    ).to(tl.int32)

                    lo_idx = byte_raw & 0xF
                    hi_idx = (byte_raw >> 4) & 0xF
                    pair_key = lo_idx * N_CENTROIDS + hi_idx

                    pair_slot = tl.arange(0, 2)
                    c_pair = tl.load(
                        Pair_lut_ptr + pair_key[:, :, None] * 2 + pair_slot[None, None, :],
                        mask=(kv_mask_1d[:, None, None] & byte_mask[None, :, None]),
                        other=0.0,
                    )
                    c_vals = tl.reshape(c_pair, [TILE_SIZE, BLOCK_D])
                elif MSE_BITS == 4:
                    half_idx = d_offs // 2
                    nibble_shift = (d_offs % 2) * 4
                    mse_addrs = slot_bases[:, None] + half_idx[None, :]
                    mse_raw = tl.load(
                        KV_cache_ptr + mse_addrs,
                        mask=kv_mask_1d[:, None] & d_mask[None, :],
                        other=0,
                    ).to(tl.int32)
                    mse_idx = (mse_raw >> nibble_shift[None, :]) & 0xF
                    c_vals = tl.load(
                        Centroids_ptr + mse_idx,
                        mask=kv_mask_1d[:, None] & d_mask[None, :],
                        other=0.0,
                    )
                else:
                    mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
                    mse_raw0 = tl.load(
                        KV_cache_ptr + mse_addrs0,
                        mask=kv_mask_1d[:, None] & d_mask[None, :],
                        other=0,
                    ).to(tl.int32)
                    mse_raw1 = tl.load(
                        KV_cache_ptr + mse_addrs0 + 1,
                        mask=kv_mask_1d[:, None] & d_mask[None, :],
                        other=0,
                    ).to(tl.int32)
                    raw16 = mse_raw0 | (mse_raw1 << 8)
                    mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask_val
                    c_vals = tl.load(
                        Centroids_ptr + mse_idx,
                        mask=kv_mask_1d[:, None] & d_mask[None, :],
                        other=0.0,
                    )

                if NORM_CORRECTION:
                    c_norm_sq = tl.sum(
                        tl.where(d_mask[None, :], c_vals * c_vals, 0.0),
                        axis=1,
                    )
                    c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                    c_vals = c_vals * c_inv_norm[:, None]

                norm_bases = slot_bases + MSE_BYTES
                n_lo = tl.load(
                    KV_cache_ptr + norm_bases,
                    mask=kv_mask_1d,
                    other=0,
                ).to(tl.uint16)
                n_hi = tl.load(
                    KV_cache_ptr + norm_bases + 1,
                    mask=kv_mask_1d,
                    other=0,
                ).to(tl.uint16)
                vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

                K_recon = c_vals * vec_norms[:, None]
                K_T = tl.trans(K_recon)
                S = QK_SCALE * tl.dot(Q.to(tl.float16), K_T.to(tl.float16))

            S = tl.where(kv_mask_1d[None, :], S, float("-inf"))

            m_j = tl.maximum(M, tl.max(S, axis=1))
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
            P = tl.math.exp2(S - m_j[:, None])
            l_j = tl.sum(P, axis=1)
            alpha = tl.math.exp2(M - m_j)

            acc = acc * alpha[:, None]
            L = L * alpha + l_j
            M = m_j

            val_bases = slot_bases + KPS

            if VQB == 3:
                val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
                val_raw0 = tl.load(
                    KV_cache_ptr + val_addrs0,
                    mask=kv_mask_1d[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                val_raw1 = tl.load(
                    KV_cache_ptr + val_addrs0 + 1,
                    mask=kv_mask_1d[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                raw16 = val_raw0 | (val_raw1 << 8)
                v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)

                sc_bases = val_bases + VAL_DATA_BYTES
                sc_lo = tl.load(
                    KV_cache_ptr + sc_bases, mask=kv_mask_1d, other=0,
                ).to(tl.uint16)
                sc_hi = tl.load(
                    KV_cache_ptr + sc_bases + 1, mask=kv_mask_1d, other=0,
                ).to(tl.uint16)
                v_scales = (
                    (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                )
                zr_lo = tl.load(
                    KV_cache_ptr + sc_bases + 2, mask=kv_mask_1d, other=0,
                ).to(tl.uint16)
                zr_hi = tl.load(
                    KV_cache_ptr + sc_bases + 3, mask=kv_mask_1d, other=0,
                ).to(tl.uint16)
                v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                V = v_idx * v_scales[:, None] + v_zeros[:, None]
            else:  # VQB == 4
                # PR#40396 halves the loads here with tl.interleave(lo, hi).
                # It does not reproduce the stock kernel: at vqb=4 it lands at
                # rel diff 2.15e-2 where this extraction reads 7.4e-6 (job
                # 122969, A/B on identical inputs). k2v3 is vqb=3 so the path
                # is dead for our arm either way; keeping the correct one.
                vb_idx = d_offs // 2
                vb_shift = (d_offs % 2) * 4
                v_raw = tl.load(
                    KV_cache_ptr + val_bases[:, None] + vb_idx[None, :],
                    mask=kv_mask_1d[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                v_idx = ((v_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

                sc_bases = val_bases + VAL_DATA_BYTES
                sc_lo = tl.load(
                    KV_cache_ptr + sc_bases, mask=kv_mask_1d, other=0,
                ).to(tl.uint16)
                sc_hi = tl.load(
                    KV_cache_ptr + sc_bases + 1, mask=kv_mask_1d, other=0,
                ).to(tl.uint16)
                v_scales = (
                    (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                )
                zr_lo = tl.load(
                    KV_cache_ptr + sc_bases + 2, mask=kv_mask_1d, other=0,
                ).to(tl.uint16)
                zr_hi = tl.load(
                    KV_cache_ptr + sc_bases + 3, mask=kv_mask_1d, other=0,
                ).to(tl.uint16)
                v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                V = v_idx * v_scales[:, None] + v_zeros[:, None]

            acc += tl.dot(P.to(tl.float16), V.to(tl.float16))

        safe_L = tl.where(L > 0.0, L, 1.0)
        one_over_L = 1.0 / safe_L
        acc = acc * one_over_L[:, None]
        lse = M * LN2 + tl.log(safe_L)

        out_addrs = (
            bid * stride_mid_b
            + q_head_offs[:, None] * stride_mid_h
            + sid * stride_mid_s
            + d_offs[None, :]
        )
        tl.store(
            Mid_o_ptr + out_addrs, acc, mask=q_mask[:, None] & d_mask[None, :],
        )
        lse_addrs = (
            bid * stride_mid_b + q_head_offs * stride_mid_h + sid * stride_mid_s + HEAD_DIM
        )
        tl.store(Mid_o_ptr + lse_addrs, lse, mask=q_mask)

    def _v2_decode_attention(
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        scale: float,
        mse_bits: int,
        key_packed_size: int,
        value_quant_bits: int,
        key_fp8: bool = False,
        norm_correction: bool = False,
        PiT: torch.Tensor | None = None,
        mid_o_buf: torch.Tensor | None = None,
        output_buf: torch.Tensor | None = None,
        lse_buf: torch.Tensor | None = None,
        buf_holder: Any = None,
        max_num_kv_splits: int = 32,
    ) -> torch.Tensor:
        B, Hq, D = query.shape
        Hk = kv_cache.shape[2]
        block_size = kv_cache.shape[1]
        padded_slot = kv_cache.shape[3]
        max_num_blocks = block_table.shape[1]
        n_centroids = centroids.shape[0]
        kv_group_size = Hq // Hk
        device = query.device

        cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

        if key_fp8:
            q_rot = query.float().contiguous()
        else:
            q_float = query.float()
            if PiT is None:
                PiT = Pi.T.contiguous()
            q_rot = (q_float @ PiT).contiguous()

        BLOCK_M = max(16, triton.next_power_of_2(kv_group_size))
        TILE_SIZE = 16
        NUM_KV_SPLITS = max_num_kv_splits

        if (
            mid_o_buf is not None
            and mid_o_buf.shape[0] >= B
            and mid_o_buf.shape[2] >= NUM_KV_SPLITS
        ):
            mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
        else:
            mid_o = torch.empty(
                B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=device,
            )
            if buf_holder is not None:
                buf_holder._tq_mid_o_buf = mid_o

        fp8_e4b15 = _use_fp8_e4b15(device.index or 0)

        use_pair_lut = mse_bits == 4 and not key_fp8
        pair_lut = _build_pair_lut(centroids) if use_pair_lut else centroids

        grid = (B, Hk, NUM_KV_SPLITS)
        _tq_decode_stage1_v2[grid](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            pair_lut,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            NUM_Q_HEADS=Hq,
            NUM_KV_HEADS=Hk,
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            PADDED_SLOT=padded_slot,
            MAX_NUM_BLOCKS=max_num_blocks,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=mse_bits,
            MSE_BYTES=cfg["mse_bytes"],
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=cfg["val_data_bytes"],
            N_CENTROIDS=n_centroids,
            ATTN_SCALE=scale,
            BLOCK_D=cfg["BLOCK_D"],
            TILE_SIZE=TILE_SIZE,
            BLOCK_M=BLOCK_M,
            KEY_FP8=1 if key_fp8 else 0,
            NORM_CORRECTION=1 if norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
            USE_PAIR_LUT=1 if use_pair_lut else 0,
            num_warps=4,
            num_stages=2,
        )

        # Output buffer handling follows OUR installed v1 launcher, not the
        # draft PR's: the backend pre-allocates output_buf in the model dtype
        # and stage2 writes it directly (OUTPUT_FP16), whereas the PR assumed
        # an fp32 buffer plus a trailing cast.
        out_dtype = query.dtype
        if (
            output_buf is not None
            and output_buf.shape[0] >= B
            and output_buf.dtype == out_dtype
        ):
            output = output_buf[:B, :Hq, :D]
        else:
            output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
            if buf_holder is not None:
                buf_holder._tq_output_buf = output
        if lse_buf is not None and lse_buf.shape[0] >= B:
            lse = lse_buf[:B, :Hq]
        else:
            lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
            if buf_holder is not None:
                buf_holder._tq_lse_buf = lse

        grid2 = (B, Hq)
        _fwd_kernel_stage2[grid2](
            mid_o,
            output,
            lse,
            seq_lens,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            BLOCK_DV=cfg["BLOCK_D"],
            Lv=D,
            OUTPUT_FP16=1 if out_dtype == torch.float16 else 0,
            num_warps=4,
            num_stages=2,
        )
        return output

    _tqa.triton_turboquant_decode_attention = _v2_decode_attention
    _applied = True
    if verbose:
        print("[tq-v2] k2v3 decode stage-1 replaced with ported v2 kernel "
              "(tensor-core tl.dot, vectorized pair LUT, exp2); numerics "
              "differ from the stock scalar kernel, verify before trusting "
              "eval/training numbers (in-process patch; shared install "
              "untouched)", flush=True)
    return True
