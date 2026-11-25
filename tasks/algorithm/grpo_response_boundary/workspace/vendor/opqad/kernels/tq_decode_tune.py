"""Opt-in: tune the TurboQuant k2v3 decode kernel's stage-1 launch config.

k2v3 AIME evals and training rollout run 2.2x+ slower than OSCAR's fast path
for the same protocol (job 120605: 1:39:11 vs job 120601/120602: still
running past 3:38).  A fork-agent investigation this session ruled out the
two fixes that helped OSCAR (full decode CUDA graphs, per-request staging):
k2v3's backend already declares `AttentionCGSupport.UNIFORM_BATCH`
(`turboquant_attn.py:201`) and gets `FULL_AND_PIECEWISE` by default, and it
has no staging arena to patch.  The actual bottleneck is inside the one
Triton kernel launch that does the real work per decode step:

    vllm/v1/attention/ops/triton_turboquant_decode.py:537
    (triton_turboquant_decode_attention), stage-1 launch ~603-638

which hardcodes

    BLOCK_KV = 4
    num_warps=1, num_stages=1

4 KV tokens per tile is far smaller than typical paged-attention tiling
(16-64), and warps=1/stages=1 leaves almost no instruction-level
parallelism or memory-latency hiding.  This looks like a correctness-first
default that was never perf-tuned, unlike OSCAR's kernel
(`oscar_cudagraph_patch.py`), which has a documented, benchmarked config.

What this changes
------------------
Replaces `vllm.v1.attention.backends.turboquant_attn.triton_turboquant_decode_attention`
(the name is REBOUND into that module's namespace by its own `from ... import`,
so the patch target must be that module's attribute, not the ops module's --
patching the origin module leaves the backend's already-bound reference
untouched) with a function whose body is otherwise identical to vLLM's, but
whose stage-1 launch reads BLOCK_KV / num_warps / num_stages from
QAT_TQ_BLOCK_KV / QAT_TQ_NUM_WARPS / QAT_TQ_NUM_STAGES instead of the
hardcoded 4 / 1 / 1.  Stage 2 (`_fwd_kernel_stage2`, warps=4/stages=2) is
untouched -- the fork agent's report did not flag it, and this stays scoped
to the one launch that is suspiciously small.

This is a pure launch-config change: same kernel, same reduction, same
inputs, only the tiling/occupancy of the GPU launch differs.  It should not
change which tokens are produced, but tiling changes CAN in principle
perturb floating-point reduction order (splitting the KV axis differently
changes the order values are summed in), so any non-default config must be
verified token-identical under greedy decoding before being trusted
(see experiments/2026-09-22/tq_decode_tune/).

Default off
-----------
Gated on QAT_OSCAR_FULL_CUDAGRAPH's sibling flag, QAT_TQ_DECODE_TUNE=1. Env
vars QAT_TQ_BLOCK_KV/QAT_TQ_NUM_WARPS/QAT_TQ_NUM_STAGES default to today's
hardcoded values (4/1/1) via opqad/utils/envreg.py, so even if this module is
imported and applied without further configuration, it is a numerical and
performance no-op.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from opqad.utils.envreg import env_int

ENV_FLAG = "QAT_TQ_DECODE_TUNE"

_applied = False


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") not in ("", "0", "false", "False")


def build_launcher(stage1_kernel=None):
    """Build a drop-in replacement for `triton_turboquant_decode_attention`.

    Raises on import failure (no TurboQuant backend in this process);
    callers decide whether that is fatal.  `stage1_kernel` overrides which
    stage-1 kernel is launched, so a module that only rewrites the kernel
    body -- opqad/kernels/tq_decode_lut.py -- reuses this launcher instead of
    carrying a third copy of it that would drift.
    """
    from vllm.v1.attention.ops.triton_decode_attention import (
        _fwd_kernel_stage2,
    )
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _get_layout,
        _tq_decode_stage1,
        _use_fp8_e4b15,
    )

    stage1 = _tq_decode_stage1 if stage1_kernel is None else stage1_kernel
    block_kv = env_int("QAT_TQ_BLOCK_KV")
    num_warps = env_int("QAT_TQ_NUM_WARPS")
    num_stages = env_int("QAT_TQ_NUM_STAGES")

    def _tuned_decode_attention(
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
        kv_group_size = Hq // Hk
        device = query.device

        cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

        if key_fp8:
            q_rot = query.contiguous()
        else:
            q_float = query.float()
            if PiT is None:
                PiT = Pi.T.contiguous()
            q_rot = (q_float @ PiT).contiguous()

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
        grid = (B, Hq, NUM_KV_SPLITS)
        stage1[grid](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
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
            NUM_KV_HEADS=Hk,
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=mse_bits,
            MSE_BYTES=cfg["mse_bytes"],
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=cfg["val_data_bytes"],
            ATTN_SCALE=scale,
            BLOCK_D=cfg["BLOCK_D"],
            BLOCK_KV=block_kv,
            KEY_FP8=1 if key_fp8 else 0,
            NORM_CORRECTION=1 if norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
            num_warps=num_warps,
            num_stages=num_stages,
        )

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

    return _tuned_decode_attention


def apply(verbose: bool = True) -> bool:
    """Patch the in-process k2v3 decode kernel launch config.

    Returns True if applied.  Safe to call when vLLM has no TurboQuant
    backend loaded (an OSCAR or BF16 arm): the import fails and this returns
    False rather than raising.  Must be called before the first decode call,
    i.e. before LLM(...) or, for training, before rollout starts.
    """
    global _applied
    if _applied or not enabled():
        return False
    try:
        import vllm.v1.attention.backends.turboquant_attn as _tqa
        launcher = build_launcher()
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[tq-tune] not applied: {type(exc).__name__}: {exc}",
                  flush=True)
        return False

    _tqa.triton_turboquant_decode_attention = launcher
    _applied = True
    if verbose:
        print(f"[tq-tune] k2v3 decode stage-1 retuned: "
              f"BLOCK_KV={env_int('QAT_TQ_BLOCK_KV')} "
              f"num_warps={env_int('QAT_TQ_NUM_WARPS')} "
              f"num_stages={env_int('QAT_TQ_NUM_STAGES')} "
              "(in-process patch; shared install untouched)", flush=True)
    return True
