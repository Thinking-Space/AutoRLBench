"""Hopper (SM90) CuTe DSL fused log-prob kernel.

This module implements fused lm_head + log-prob computation using CuTe DSL for
Hopper (SM90, H800) GPUs. It fuses the classifier projection ``Z = X @ W.T``
with the online logsumexp so the full ``[T, V]`` logits never materialize in HBM.

Two fused_kernels are provided:

1. ``cute_fused_logprob_forward_tile``: Forward pass - computes logits, online
   logsumexp update, and target logit gathering for one vocab tile.

2. ``cute_fused_logprob_backward_tile``: Backward pass - recomputes logits,
   builds softmax probs, and accumulates gradients for hidden, weight, and bias.

Both fused_kernels compute rank-local partials only. The Python host performs all-reduce
across TP ranks to stitch the results.

Requirements
------------
Hopper (compute capability 9.0), BF16/FP16 input/weight, FP32 accumulation.
The fused_kernels handle ragged shapes via padding internally.

References
----------
- CuTe DSL CTA norm example: warp shuffle + smem reduce patterns
- Liger SM90 fused-linear-CE: production reference for fused softmax
- Miles fused_logprob.py: PyTorch semantics this kernel must match exactly
"""

from __future__ import annotations

import math
from typing import Optional

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# =============================================================================
# Warp-level and CTA-level reduction primitives
# =============================================================================


@cute.jit
def warp_reduce_max(val: cutlass.Float32, reduce_size: int = 32) -> cutlass.Float32:
    """Reduce values within a warp using butterfly shuffle (max).

    Args:
        val: Per-thread value to reduce
        reduce_size: Number of threads participating (must be power of 2, <= 32)

    Returns:
        The reduced max value (broadcast to all threads in the warp)
    """
    iters = int(math.log2(reduce_size))
    for i in range(iters):
        other = cute.arch.shuffle_sync_bfly(val, offset=1 << i)
        val = cute.arch.fmax(val, other)
    return val


@cute.jit
def warp_reduce_sum(val: cutlass.Float32, reduce_size: int = 32) -> cutlass.Float32:
    """Reduce values within a warp using butterfly shuffle (sum).

    Args:
        val: Per-thread value to reduce
        reduce_size: Number of threads participating (must be power of 2, <= 32)

    Returns:
        The reduced sum value (broadcast to all threads in the warp)
    """
    iters = int(math.log2(reduce_size))
    for i in range(iters):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


@cute.jit
def cta_reduce_max(
    val: cutlass.Float32,
    smem_buf: cute.Tensor,
    warps_per_cta: int,
    tidx: cutlass.Int32,
) -> cutlass.Float32:
    """CTA-level max reduction using shared memory.

    Performs a two-level reduction: first within each warp using shuffle, then
    across warps using shared memory.

    Args:
        val: Per-thread value to reduce
        smem_buf: Shared memory buffer of size [warps_per_cta + 1]
        warps_per_cta: Number of warps in the CTA
        tidx: Thread index within CTA

    Returns:
        The reduced max value (broadcast to all threads)
    """
    warp_id = tidx >> 5
    lane_id = tidx & 31

    # Warp-level reduce first
    val = warp_reduce_max(val)

    # Lane 0 of each warp writes to shared memory
    if lane_id == 0:
        smem_buf[warp_id] = val
    cute.arch.sync_threads()

    # Warp 0 does final reduction
    if warp_id == 0:
        val = smem_buf[lane_id] if lane_id < warps_per_cta else cutlass.Float32(float("-inf"))
        val = warp_reduce_max(val)
        smem_buf[warps_per_cta] = val
    cute.arch.sync_threads()

    return smem_buf[warps_per_cta]


@cute.jit
def cta_reduce_sum(
    val: cutlass.Float32,
    smem_buf: cute.Tensor,
    warps_per_cta: int,
    tidx: cutlass.Int32,
) -> cutlass.Float32:
    """CTA-level sum reduction using shared memory.

    Performs a two-level reduction: first within each warp using shuffle, then
    across warps using shared memory.

    Args:
        val: Per-thread value to reduce
        smem_buf: Shared memory buffer of size [warps_per_cta + 1]
        warps_per_cta: Number of warps in the CTA
        tidx: Thread index within CTA

    Returns:
        The reduced sum value (broadcast to all threads)
    """
    warp_id = tidx >> 5
    lane_id = tidx & 31

    # Warp-level reduce first
    val = warp_reduce_sum(val)

    # Lane 0 of each warp writes to shared memory
    if lane_id == 0:
        smem_buf[warp_id] = val
    cute.arch.sync_threads()

    # Warp 0 does final reduction
    if warp_id == 0:
        val = smem_buf[lane_id] if lane_id < warps_per_cta else cutlass.Float32(0.0)
        val = warp_reduce_sum(val)
        smem_buf[warps_per_cta] = val
    cute.arch.sync_threads()

    return smem_buf[warps_per_cta]


# =============================================================================
# Forward kernel: fused GEMM + online logsumexp + target gather
# =============================================================================


class CuteFusedLogProbForwardTile:
    """CuTe DSL kernel for one forward vocab tile.

    Computes:
        logits = hidden @ weight.T (+ bias) * inv_t
        Online logsumexp update: max_new, sum_exp, target_logit

    This matches the semantics of _fused_logprob_forward_tile_impl in
    miles/backends/training_utils/loss_hub/fused_logprob.py lines 317-362.

    Each CTA processes one row (token) and computes the contribution from one
    vocab tile. The online logsumexp accumulators are threaded through the
    vocab loop by the Python host.
    """

    def __init__(
        self,
        seq_chunk: int,
        tile_vocab: int,
        hidden_dim: int,
        threads_per_cta: int = 256,
        has_bias: bool = True,
    ):
        """Initialize forward tile kernel.

        Args:
            seq_chunk: Number of tokens in this chunk
            tile_vocab: Number of vocab positions in this tile
            hidden_dim: Hidden dimension size
            threads_per_cta: Threads per CTA (must be multiple of 32)
            has_bias: Whether bias is present
        """
        self.seq_chunk = seq_chunk
        self.tile_vocab = tile_vocab
        self.hidden_dim = hidden_dim
        self.threads_per_cta = threads_per_cta
        self.warps_per_cta = threads_per_cta // 32
        self.has_bias = has_bias

        # Vocab-loop stride. Must equal ``threads_per_cta`` so the
        # ``v_idx = v_idx_base + tidx`` mapping covers every column exactly once
        # (each v_idx hit by exactly one thread). A smaller stride (e.g. 8 with
        # 256 threads) double-counts columns: v_idx 0..7 once, 8..15 twice, ...
        # up to 8x -- which silently inflates sum_exp / target_logit.
        # See docs/developer/fused_kernel_debug_log.md.
        self.elems_per_thread = 8

    @cute.jit
    def __call__(
        self,
        mMaxNew: cute.Tensor,
        mSumExp: cute.Tensor,
        mTargetLogit: cute.Tensor,
        mHidden: cute.Tensor,
        mWeight: cute.Tensor,
        mBias: Optional[cute.Tensor],
        mTargets: cute.Tensor,
        mMaxOld: cute.Tensor,
        mSumExpOld: cute.Tensor,
        mTargetLogitOld: cute.Tensor,
        mRowIdx: cute.Tensor,
        inv_t: cutlass.Float32,
        v_start: cutlass.Int32,
        v_off: cutlass.Int32,
        v_off_end: cutlass.Int32,
        v_local: cutlass.Int32,
    ):
        """Launch the forward tile kernel.

        Args:
            mMaxNew: Output [seq_chunk, 1] - updated max values
            mSumExp: Output [seq_chunk, 1] - updated sum of exponentials
            mTargetLogit: Output [seq_chunk, 1] - updated target logits
            mHidden: Input [seq_chunk, hidden_dim]
            mWeight: Input [tile_vocab, hidden_dim]
            mBias: Input [tile_vocab, 1] or None
            mTargets: Input [seq_chunk, 1] - global token ids
            mMaxOld: Input [seq_chunk, 1] - running max from previous tiles
            mSumExpOld: Input [seq_chunk, 1] - running sum_exp from previous tiles
            mTargetLogitOld: Input [seq_chunk, 1] - running target_logit from previous tiles
            mRowIdx: Input [seq_chunk, 1] - row indices (unused, for API compat)
            inv_t: 1.0 / temperature
            v_start: Global vocab start index for this rank
            v_off: Vocab offset within this rank's shard
            v_off_end: Vocab end offset within this rank's shard
            v_local: Total vocab size for this rank
        """
        self.kernel(
            mMaxNew,
            mSumExp,
            mTargetLogit,
            mHidden,
            mWeight,
            mBias,
            mTargets,
            mMaxOld,
            mSumExpOld,
            mTargetLogitOld,
            mRowIdx,
            inv_t,
            v_start,
            v_off,
            v_off_end,
            v_local,
        ).launch(
            grid=[self.seq_chunk, 1, 1],
            block=[self.threads_per_cta, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mMaxNew: cute.Tensor,
        mSumExp: cute.Tensor,
        mTargetLogit: cute.Tensor,
        mHidden: cute.Tensor,
        mWeight: cute.Tensor,
        mBias: Optional[cute.Tensor],
        mTargets: cute.Tensor,
        mMaxOld: cute.Tensor,
        mSumExpOld: cute.Tensor,
        mTargetLogitOld: cute.Tensor,
        mRowIdx: cute.Tensor,
        inv_t: cutlass.Float32,
        v_start: cutlass.Int32,
        v_off: cutlass.Int32,
        v_off_end: cutlass.Int32,
        v_local: cutlass.Int32,
    ):
        """Forward tile kernel - one CTA per row (token).

        Each CTA computes:
        1. logits[row, :] = hidden[row, :] @ weight.T + bias (then * inv_t)
        2. chunk_max = max(logits)
        3. max_new = max(max_old, chunk_max)
        4. sum_exp = sum_exp_old * exp(max_old - max_new) + sum(exp(logits - max_new))
        5. target_logit += logits[target_local] if target is in this chunk
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()  # row index

        # Allocate shared memory for reductions
        smem = cutlass.utils.SmemAllocator()
        reduce_buf = smem.allocate_tensor(cutlass.Float32, self.warps_per_cta + 1)

        # Load max_old, sum_exp_old, target_logit_old for this row
        max_old = mMaxOld[bidx, 0].to(cutlass.Float32)
        sum_exp_old = mSumExpOld[bidx, 0].to(cutlass.Float32)
        target_logit_old = mTargetLogitOld[bidx, 0].to(cutlass.Float32)
        target_global = mTargets[bidx, 0]

        # Each thread computes logits for a subset of vocab positions
        local_max = cutlass.Float32(float("-inf"))
        local_sum_exp = cutlass.Float32(0.0)
        local_target_logit = cutlass.Float32(0.0)

        # Check if target is in this rank's range and this chunk
        in_rank = (target_global >= v_start) & (target_global < v_start + v_local)
        in_chunk = in_rank & (target_global >= v_start + v_off) & (target_global < v_start + v_off_end)
        target_local = target_global - v_start - v_off

        for v_idx_base in range(0, self.tile_vocab, self.threads_per_cta):
            v_idx = v_idx_base + tidx
            if v_idx < self.tile_vocab:
                # Compute logit = dot(hidden[row], weight[v_idx]) + bias[v_idx]
                logit = cutlass.Float32(0.0)
                for h_idx in range(self.hidden_dim):
                    h_val = mHidden[bidx, h_idx].to(cutlass.Float32)
                    w_val = mWeight[v_idx, h_idx].to(cutlass.Float32)
                    logit = logit + h_val * w_val

                if cutlass.const_expr(self.has_bias):
                    logit = logit + mBias[v_idx, 0].to(cutlass.Float32)

                logit = logit * inv_t

                # Update local max (online logsumexp: rescale prior partial)
                prev_max = local_max
                local_max = cute.arch.fmax(local_max, logit)
                # Rescale the running sum_exp by exp(prev_max - local_max) so the
                # accumulator stays expressed as exp(logit_i - local_max).
                # (Online softmax: each new max forces a rescale of prior terms.)
                local_sum_exp = local_sum_exp * cute.exp(prev_max - local_max)

                # Check if this is the target position
                is_target = in_chunk & (v_idx == target_local)
                if is_target:
                    local_target_logit = logit

                # Accumulate exp(logit - local_max) (already shifted to current max)
                local_sum_exp = local_sum_exp + cute.exp(logit - local_max)

        # CTA-level reduction for max
        chunk_max = cta_reduce_max(local_max, reduce_buf, self.warps_per_cta, tidx)

        # Compute max_new = max(max_old, chunk_max)
        max_new = cute.arch.fmax(max_old, chunk_max)

        # local_sum_exp is sum_i exp(logit_i - local_max). Rescale to
        # exp(logit_i - max_new) via exp(local_max - max_new). BUG FIX: the old
        # code summed exp(logit_i) (unscaled) and then multiplied by one
        # rescale factor -- but exp(logit_i) * exp(local_max - max_new) !=
        # exp(logit_i - max_new) unless local_max == logit_i for all i (it is
        # only the running max, so the product double-shifts non-max logits).
        # See docs/developer/fused_kernel_debug_log.md.
        rescale = cute.exp(local_max - max_new)
        local_sum_exp = local_sum_exp * rescale

        # CTA-level reduction for sum_exp
        chunk_sum_exp = cta_reduce_sum(local_sum_exp, reduce_buf, self.warps_per_cta, tidx)

        # Update sum_exp: sum_exp_new = sum_exp_old * exp(max_old - max_new) + chunk_sum_exp
        rescale_old = cute.exp(max_old - max_new)
        sum_exp_new = sum_exp_old * rescale_old + chunk_sum_exp

        # CTA-level reduction for target_logit (only one thread should have it)
        target_logit_chunk = cta_reduce_sum(local_target_logit, reduce_buf, self.warps_per_cta, tidx)
        target_logit_new = target_logit_old + target_logit_chunk

        # Thread 0 writes outputs
        if tidx == 0:
            mMaxNew[bidx, 0] = max_new.to(mMaxNew.element_type)
            mSumExp[bidx, 0] = sum_exp_new.to(mSumExp.element_type)
            mTargetLogit[bidx, 0] = target_logit_new.to(mTargetLogit.element_type)


# =============================================================================
# Backward kernel: recompute logits + softmax probs + gradient accumulation
# =============================================================================


class CuteFusedLogProbBackwardTile:
    """CuTe DSL kernel for one backward vocab tile.

    Computes:
        logits = hidden @ weight.T (+ bias) * inv_t
        probs = exp(logits - logz)
        grad_logits = -grad * probs + grad * delta(target)
        grad_hidden = grad_logits @ weight
        grad_weight = grad_logits.T @ hidden
        grad_bias = sum(grad_logits, dim=0) if bias is not None

    This matches the semantics of _fused_logprob_backward_tile_impl in
    miles/backends/training_utils/loss_hub/fused_logprob.py lines 524-571.

    Each CTA processes one row (token) and computes the gradient contribution
    from one vocab tile. The Python host accumulates across vocab tiles and
    performs all-reduce across TP ranks.
    """

    def __init__(
        self,
        seq_chunk: int,
        tile_vocab: int,
        hidden_dim: int,
        threads_per_cta: int = 256,
        has_bias: bool = True,
    ):
        """Initialize backward tile kernel.

        Args:
            seq_chunk: Number of tokens in this chunk
            tile_vocab: Number of vocab positions in this tile
            hidden_dim: Hidden dimension size
            threads_per_cta: Threads per CTA (must be multiple of 32)
            has_bias: Whether bias is present
        """
        self.seq_chunk = seq_chunk
        self.tile_vocab = tile_vocab
        self.hidden_dim = hidden_dim
        self.threads_per_cta = threads_per_cta
        self.warps_per_cta = threads_per_cta // 32
        self.has_bias = has_bias

        self.elems_per_thread = 8

    @cute.jit
    def __call__(
        self,
        mGradHidden: cute.Tensor,
        mGradWeight: cute.Tensor,
        mGradBias: Optional[cute.Tensor],
        mHidden: cute.Tensor,
        mWeight: cute.Tensor,
        mBias: Optional[cute.Tensor],
        mTargets: cute.Tensor,
        mRowIdx: cute.Tensor,
        mGradChunk: cute.Tensor,
        mLogzChunk: cute.Tensor,
        inv_t: cutlass.Float32,
        v_start: cutlass.Int32,
        v_off: cutlass.Int32,
        v_off_end: cutlass.Int32,
        v_local: cutlass.Int32,
    ):
        """Launch the backward tile kernel."""
        self.kernel(
            mGradHidden,
            mGradWeight,
            mGradBias,
            mHidden,
            mWeight,
            mBias,
            mTargets,
            mRowIdx,
            mGradChunk,
            mLogzChunk,
            inv_t,
            v_start,
            v_off,
            v_off_end,
            v_local,
        ).launch(
            grid=[self.seq_chunk, 1, 1],
            block=[self.threads_per_cta, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mGradHidden: cute.Tensor,
        mGradWeight: cute.Tensor,
        mGradBias: Optional[cute.Tensor],
        mHidden: cute.Tensor,
        mWeight: cute.Tensor,
        mBias: Optional[cute.Tensor],
        mTargets: cute.Tensor,
        mRowIdx: cute.Tensor,
        mGradChunk: cute.Tensor,
        mLogzChunk: cute.Tensor,
        inv_t: cutlass.Float32,
        v_start: cutlass.Int32,
        v_off: cutlass.Int32,
        v_off_end: cutlass.Int32,
        v_local: cutlass.Int32,
    ):
        """Backward tile kernel - one CTA per row (token).

        Each CTA computes:
        1. Recompute logits[row, :] = hidden[row, :] @ weight.T + bias (then * inv_t)
        2. probs = exp(logits - logz)
        3. grad_logits = -grad * probs, then add grad to target position
        4. Accumulate grad_hidden, grad_weight, grad_bias
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()  # row index

        # Allocate shared memory for reductions.
        smem = cutlass.utils.SmemAllocator()
        reduce_buf = smem.allocate_tensor(cutlass.Float32, self.warps_per_cta + 1)
        # Per-thread grad_hidden accumulator. A Python list of staged values indexed by a
        # loop var is rejected by the DSL (PHASE_DYNAMIC_INDEX); a shared-memory tensor
        # indexed by ``tidx * hidden_dim + h_idx`` is the supported form (cf. cta_norm's
        # ``acc[warp_id]``). Each thread owns a private row; the CTA reduce below folds
        # the rows together. See docs/developer/fused_kernel_debug_log.md.
        grad_hidden_buf = smem.allocate_tensor(
            cutlass.Float32, self.threads_per_cta * self.hidden_dim
        )
        for h_idx in range(self.hidden_dim):
            grad_hidden_buf[tidx * self.hidden_dim + h_idx] = cutlass.Float32(0.0)
        cute.arch.sync_threads()

        # Load grad and logz for this row
        grad_val = mGradChunk[bidx, 0].to(cutlass.Float32)
        logz_val = mLogzChunk[bidx, 0].to(cutlass.Float32)
        target_global = mTargets[bidx, 0]

        # Base iterators for scattered atomic adds (grad_weight / grad_bias / grad_hidden).
        # ``atomic_add`` takes ``(ptr, val)``; ``mGradW[v_idx, h_idx]`` returns a scalar
        # value (no ``.iterator``), so we advance the tensor's base iterator by the flat
        # offset instead -- the supported way to atomically update a 2D element.
        grad_weight_base = mGradWeight.iterator
        grad_bias_base = mGradBias.iterator if cutlass.const_expr(self.has_bias) else None

        # Check if target is in this rank's range and this chunk
        in_rank = (target_global >= v_start) & (target_global < v_start + v_local)
        in_chunk = in_rank & (target_global >= v_start + v_off) & (target_global < v_start + v_off_end)
        target_local = target_global - v_start - v_off

        # Each thread accumulates gradients for a subset of vocab positions
        # We need to accumulate grad_weight and grad_bias across vocab,
        # and grad_hidden across vocab (then reduce across threads)

        for v_idx_base in range(0, self.tile_vocab, self.threads_per_cta):
            v_idx = v_idx_base + tidx
            if v_idx < self.tile_vocab:
                # Recompute logit
                logit = cutlass.Float32(0.0)
                for h_idx in range(self.hidden_dim):
                    h_val = mHidden[bidx, h_idx].to(cutlass.Float32)
                    w_val = mWeight[v_idx, h_idx].to(cutlass.Float32)
                    logit = logit + h_val * w_val

                if cutlass.const_expr(self.has_bias):
                    logit = logit + mBias[v_idx, 0].to(cutlass.Float32)

                logit = logit * inv_t

                # Compute prob = exp(logit - logz)
                prob = cute.exp(logit - logz_val)

                # grad_logit = -grad * prob
                grad_logit = -grad_val * prob

                # Add grad to target position
                is_target = in_chunk & (v_idx == target_local)
                if is_target:
                    grad_logit = grad_logit + grad_val

                # Apply inv_t chain rule
                grad_logit = grad_logit * inv_t

                # Accumulate grad_hidden into this thread's smem row:
                # grad_hidden[h] += grad_logit * weight[v_idx, h]
                for h_idx in range(self.hidden_dim):
                    w_val = mWeight[v_idx, h_idx].to(cutlass.Float32)
                    slot = tidx * self.hidden_dim + h_idx
                    grad_hidden_buf[slot] = grad_hidden_buf[slot] + grad_logit * w_val

                # Atomic add to grad_weight (across rows): base ptr + flat offset.
                for h_idx in range(self.hidden_dim):
                    h_val = mHidden[bidx, h_idx].to(cutlass.Float32)
                    off = v_idx * self.hidden_dim + h_idx
                    cute.arch.atomic_add(ptr=grad_weight_base + off, val=grad_logit * h_val)

                # Atomic add to grad_bias
                if cutlass.const_expr(self.has_bias):
                    cute.arch.atomic_add(ptr=grad_bias_base + v_idx, val=grad_logit)

        cute.arch.sync_threads()

        # Reduce grad_hidden across threads using CTA reduction
        grad_hidden_base = mGradHidden.iterator
        for h_idx in range(self.hidden_dim):
            grad_h_reduced = cta_reduce_sum(
                grad_hidden_buf[tidx * self.hidden_dim + h_idx],
                reduce_buf,
                self.warps_per_cta,
                tidx,
            )
            # Thread 0 writes to grad_hidden
            if tidx == 0:
                # Atomic add since multiple CTAs (rows) write to the same grad_hidden
                off = bidx * self.hidden_dim + h_idx
                cute.arch.atomic_add(ptr=grad_hidden_base + off, val=grad_h_reduced)


# =============================================================================
# Python wrappers matching PyTorch tile impl signatures
# =============================================================================


def cute_fused_logprob_forward_tile(
    hidden_chunk: torch.Tensor,
    weight_chunk: torch.Tensor,
    bias_chunk: Optional[torch.Tensor],
    targets_chunk: torch.Tensor,
    row_idx: torch.Tensor,
    max_old: torch.Tensor,
    sum_exp: torch.Tensor,
    target_logit: torch.Tensor,
    inv_t: float,
    v_start: int,
    v_off: int,
    v_off_end: int,
    v_local: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CuTe DSL forward tile wrapper.

    Matches _fused_logprob_forward_tile_impl signature exactly.

    Args:
        hidden_chunk: [seq_chunk, H] hidden activations
        weight_chunk: [tile_vocab, H] weight shard for this vocab tile
        bias_chunk: [tile_vocab] or None - bias shard for this vocab tile
        targets_chunk: [seq_chunk] global token ids
        row_idx: [seq_chunk] row indices (not used in kernel, kept for API compat)
        max_old: [seq_chunk] running max from previous tiles
        sum_exp: [seq_chunk] running sum_exp from previous tiles
        target_logit: [seq_chunk] running target_logit from previous tiles
        inv_t: 1.0 / temperature
        v_start: Global vocab start index for this rank
        v_off: Vocab offset within this rank's shard
        v_off_end: Vocab end offset within this rank's shard
        v_local: Total vocab size for this rank

    Returns:
        (max_new, sum_exp_new, target_logit_new) - updated accumulators
    """
    seq_chunk, hidden_dim = hidden_chunk.shape
    tile_vocab = weight_chunk.shape[0]
    has_bias = bias_chunk is not None

    # Allocate output tensors
    max_new = torch.empty_like(max_old)
    sum_exp_new = torch.empty_like(sum_exp)
    target_logit_new = torch.empty_like(target_logit)

    # Determine thread count based on problem size. The vocab-loop stride
    # inside the kernel equals ``threads_per_cta`` so the
    # ``v_idx = v_idx_base + tidx`` mapping covers every column exactly once.
    # (See docs/developer/fused_kernel_debug_log.md -- a stride smaller than
    # threads_per_cta double-counts columns and silently inflates sum_exp /
    # target_logit.)
    elems_per_thread = 8
    threads_per_cta = elems_per_thread if tile_vocab < elems_per_thread else 256

    # Create kernel instance
    kernel = CuteFusedLogProbForwardTile(
        seq_chunk=seq_chunk,
        tile_vocab=tile_vocab,
        hidden_dim=hidden_dim,
        threads_per_cta=threads_per_cta,
        has_bias=has_bias,
    )

    # Convert tensors to CuTe format
    _hidden = from_dlpack(hidden_chunk.contiguous(), assumed_align=16)
    _weight = from_dlpack(weight_chunk.contiguous(), assumed_align=16)
    _bias = from_dlpack(bias_chunk.contiguous().unsqueeze(-1), assumed_align=16) if has_bias else None
    _targets = from_dlpack(targets_chunk.contiguous().unsqueeze(-1), assumed_align=16)
    _max_old = from_dlpack(max_old.contiguous().unsqueeze(-1), assumed_align=16)
    _sum_exp_old = from_dlpack(sum_exp.contiguous().unsqueeze(-1), assumed_align=16)
    _target_logit_old = from_dlpack(target_logit.contiguous().unsqueeze(-1), assumed_align=16)
    _row_idx = from_dlpack(row_idx.contiguous().unsqueeze(-1), assumed_align=16)
    _max_new = from_dlpack(max_new.contiguous().unsqueeze(-1), assumed_align=16)
    _sum_exp_new = from_dlpack(sum_exp_new.contiguous().unsqueeze(-1), assumed_align=16)
    _target_logit_new = from_dlpack(target_logit_new.contiguous().unsqueeze(-1), assumed_align=16)

    # Compile and launch
    compiled_kernel = cute.compile(
        kernel,
        _max_new,
        _sum_exp_new,
        _target_logit_new,
        _hidden,
        _weight,
        _bias,
        _targets,
        _max_old,
        _sum_exp_old,
        _target_logit_old,
        _row_idx,
        cutlass.Float32(inv_t),
        cutlass.Int32(v_start),
        cutlass.Int32(v_off),
        cutlass.Int32(v_off_end),
        cutlass.Int32(v_local),
    )

    compiled_kernel(
        _max_new,
        _sum_exp_new,
        _target_logit_new,
        _hidden,
        _weight,
        _bias,
        _targets,
        _max_old,
        _sum_exp_old,
        _target_logit_old,
        _row_idx,
        cutlass.Float32(inv_t),
        cutlass.Int32(v_start),
        cutlass.Int32(v_off),
        cutlass.Int32(v_off_end),
        cutlass.Int32(v_local),
    )

    return max_new, sum_exp_new, target_logit_new


def cute_fused_logprob_backward_tile(
    hidden_chunk: torch.Tensor,
    weight_chunk: torch.Tensor,
    bias_chunk: Optional[torch.Tensor],
    targets_chunk: torch.Tensor,
    row_idx: torch.Tensor,
    grad_chunk: torch.Tensor,
    logz_chunk: torch.Tensor,
    inv_t: float,
    v_start: int,
    v_off: int,
    v_off_end: int,
    v_local: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """CuTe DSL backward tile wrapper.

    Matches _fused_logprob_backward_tile_impl signature exactly.

    Args:
        hidden_chunk: [seq_chunk, H] hidden activations
        weight_chunk: [tile_vocab, H] weight shard for this vocab tile
        bias_chunk: [tile_vocab] or None - bias shard for this vocab tile
        targets_chunk: [seq_chunk] global token ids
        row_idx: [seq_chunk] row indices (not used in kernel, kept for API compat)
        grad_chunk: [seq_chunk] upstream gradient (grad_logprobs)
        logz_chunk: [seq_chunk] log-sum-exp (full vocab, already stitched across TP)
        inv_t: 1.0 / temperature
        v_start: Global vocab start index for this rank
        v_off: Vocab offset within this rank's shard
        v_off_end: Vocab end offset within this rank's shard
        v_local: Total vocab size for this rank

    Returns:
        (grad_hidden_tile, grad_weight_tile, grad_bias_tile) - local partial gradients
    """
    seq_chunk, hidden_dim = hidden_chunk.shape
    tile_vocab = weight_chunk.shape[0]
    has_bias = bias_chunk is not None

    # Allocate output tensors (zero-initialized for atomic accumulation)
    grad_hidden_tile = torch.zeros((seq_chunk, hidden_dim), device=hidden_chunk.device, dtype=torch.float32)
    grad_weight_tile = torch.zeros((tile_vocab, hidden_dim), device=weight_chunk.device, dtype=torch.float32)
    grad_bias_tile = torch.zeros((tile_vocab,), device=weight_chunk.device, dtype=torch.float32) if has_bias else None

    # Determine thread count (see forward wrapper for the stride rationale)
    elems_per_thread = 8
    threads_per_cta = elems_per_thread if tile_vocab < elems_per_thread else 256

    # Create kernel instance
    kernel = CuteFusedLogProbBackwardTile(
        seq_chunk=seq_chunk,
        tile_vocab=tile_vocab,
        hidden_dim=hidden_dim,
        threads_per_cta=threads_per_cta,
        has_bias=has_bias,
    )

    # Convert tensors to CuTe format
    _hidden = from_dlpack(hidden_chunk.contiguous(), assumed_align=16)
    _weight = from_dlpack(weight_chunk.contiguous(), assumed_align=16)
    _bias = from_dlpack(bias_chunk.contiguous().unsqueeze(-1), assumed_align=16) if has_bias else None
    _targets = from_dlpack(targets_chunk.contiguous().unsqueeze(-1), assumed_align=16)
    _row_idx = from_dlpack(row_idx.contiguous().unsqueeze(-1), assumed_align=16)
    _grad_chunk = from_dlpack(grad_chunk.contiguous().unsqueeze(-1), assumed_align=16)
    _logz_chunk = from_dlpack(logz_chunk.contiguous().unsqueeze(-1), assumed_align=16)
    _grad_hidden = from_dlpack(grad_hidden_tile.contiguous(), assumed_align=16)
    _grad_weight = from_dlpack(grad_weight_tile.contiguous(), assumed_align=16)
    _grad_bias = from_dlpack(grad_bias_tile.contiguous().unsqueeze(-1), assumed_align=16) if has_bias else None

    # Compile and launch
    compiled_kernel = cute.compile(
        kernel,
        _grad_hidden,
        _grad_weight,
        _grad_bias,
        _hidden,
        _weight,
        _bias,
        _targets,
        _row_idx,
        _grad_chunk,
        _logz_chunk,
        cutlass.Float32(inv_t),
        cutlass.Int32(v_start),
        cutlass.Int32(v_off),
        cutlass.Int32(v_off_end),
        cutlass.Int32(v_local),
    )

    compiled_kernel(
        _grad_hidden,
        _grad_weight,
        _grad_bias,
        _hidden,
        _weight,
        _bias,
        _targets,
        _row_idx,
        _grad_chunk,
        _logz_chunk,
        cutlass.Float32(inv_t),
        cutlass.Int32(v_start),
        cutlass.Int32(v_off),
        cutlass.Int32(v_off_end),
        cutlass.Int32(v_local),
    )

    return grad_hidden_tile, grad_weight_tile, grad_bias_tile


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "cute_fused_logprob_forward_tile",
    "cute_fused_logprob_backward_tile",
    "CuteFusedLogProbForwardTile",
    "CuteFusedLogProbBackwardTile",
]
