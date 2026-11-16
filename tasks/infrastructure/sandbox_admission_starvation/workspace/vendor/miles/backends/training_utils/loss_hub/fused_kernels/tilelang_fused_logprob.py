"""TileLang fused logprob fused_kernels for lm_head + log-prob computation.

This module provides TileLang implementations of the fused forward and backward
passes for computing log-probabilities without materializing the full logits tensor.

The fused_kernels compute rank-local partials only. The caller is responsible for
all-reduce across TP ranks.

Reference: miles/backends/training_utils/loss_hub/fused_logprob.py
"""

from typing import Optional

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[12, 13, 14],
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    compile_flags=["-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"],
)
def _tilelang_fwd_tile_kernel(
    T_block: int,
    H: int,
    V_tile: int,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
    has_bias: bool = True,
    inv_t: float = 1.0,
):
    """TileLang kernel for forward vocab tile.

    Computes:
        logits = hidden @ weight.T (+ bias, * inv_t)
        Online logsumexp update: max_new, sum_exp
        Target gather: target_logit

    Args:
        T_block: Sequence chunk size
        H: Hidden dimension
        V_tile: Vocab tile size
        dtype: Input dtype
        accum_dtype: Accumulation dtype
        has_bias: Whether bias is present
        inv_t: Inverse temperature scaling

    Returns:
        prim_func for the kernel
    """
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    @T.prim_func
    def fwd(
        hidden_chunk: T.Tensor([T_block, H], dtype),
        weight_chunk: T.Tensor([V_tile, H], dtype),
        bias_chunk: T.Tensor([V_tile], accum_dtype) if has_bias else T.Tensor([1], accum_dtype),
        targets_chunk: T.Tensor([T_block], T.int64),
        row_idx: T.Tensor([T_block], T.int64),
        max_old: T.Tensor([T_block], accum_dtype),
        sum_exp: T.Tensor([T_block], accum_dtype),
        target_logit: T.Tensor([T_block], accum_dtype),
        v_start: T.Tensor([1], T.int32),
        v_off: T.Tensor([1], T.int32),
        v_off_end: T.Tensor([1], T.int32),
        v_local: T.Tensor([1], T.int32),
        max_new: T.Tensor([T_block], accum_dtype),
        sum_exp_new: T.Tensor([T_block], accum_dtype),
        target_logit_new: T.Tensor([T_block], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(T_block, BLOCK_M), threads=128) as (bx):
            # Shared memory for input tiles
            h_shared = T.alloc_shared([BLOCK_M, BLOCK_K], dtype)
            w_shared = T.alloc_shared([BLOCK_N, BLOCK_K], dtype)
            b_shared = T.alloc_shared([BLOCK_N], accum_dtype) if has_bias else T.alloc_shared([1], accum_dtype)

            # Fragment for logits accumulation
            logits = T.alloc_fragment([BLOCK_M, BLOCK_N], accum_dtype)
            logits_shared = T.alloc_shared([BLOCK_M, BLOCK_N], accum_dtype)

            # Fragments for reductions
            chunk_max = T.alloc_fragment([BLOCK_M], accum_dtype)
            chunk_exp = T.alloc_fragment([BLOCK_M, BLOCK_N], accum_dtype)
            chunk_sum_exp = T.alloc_fragment([BLOCK_M], accum_dtype)

            # Fragments for online accumulation
            local_max_new = T.alloc_fragment([BLOCK_M], accum_dtype)
            local_sum_exp = T.alloc_fragment([BLOCK_M], accum_dtype)
            local_target_logit = T.alloc_fragment([BLOCK_M], accum_dtype)

            # Initialize accumulators
            T.fill(local_max_new, -T.infinity(accum_dtype))
            T.fill(local_sum_exp, 0.0)
            T.fill(local_target_logit, 0.0)

            # Iterate over vocab chunks within this tile
            for v_chunk_idx in T.serial(T.ceildiv(V_tile, BLOCK_N)):
                v_chunk_start = v_chunk_idx * BLOCK_N
                v_chunk_end = T.min(v_chunk_start + BLOCK_N, V_tile)

                # Compute logits for this vocab chunk: h @ w.T
                T.clear(logits)

                # Load bias if present
                if has_bias:
                    T.copy(bias_chunk[v_chunk_start], b_shared)

                # GEMM: logits = hidden @ weight.T
                for k in T.serial(T.ceildiv(H, BLOCK_K)):
                    T.copy(hidden_chunk[bx * BLOCK_M, k * BLOCK_K], h_shared)
                    T.copy(weight_chunk[v_chunk_start, k * BLOCK_K], w_shared)
                    T.gemm(h_shared, w_shared, logits, transpose_B=True)

                # Add bias and apply inv_t
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row = bx * BLOCK_M + i
                    col = v_chunk_start + j
                    if row < T_block and col < V_tile:
                        if has_bias:
                            logits[i, j] = logits[i, j] + b_shared[j]
                        if inv_t != 1.0:
                            logits[i, j] = logits[i, j] * inv_t

                # Find max in this vocab chunk
                T.reduce_max(logits, chunk_max, dim=1, clear=True)

                # Update running max
                for i in T.Parallel(BLOCK_M):
                    row = bx * BLOCK_M + i
                    if row < T_block:
                        local_max_new[i] = T.max(local_max_new[i], chunk_max[i])

                # Compute exp(logits - max_new) and sum
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row = bx * BLOCK_M + i
                    col = v_chunk_start + j
                    if row < T_block and col < V_tile:
                        chunk_exp[i, j] = T.exp(logits[i, j] - local_max_new[i])
                    else:
                        chunk_exp[i, j] = 0.0

                T.reduce_sum(chunk_exp, chunk_sum_exp, dim=1, clear=True)

                # Accumulate sum_exp
                for i in T.Parallel(BLOCK_M):
                    row = bx * BLOCK_M + i
                    if row < T_block:
                        local_sum_exp[i] = local_sum_exp[i] + chunk_sum_exp[i]

                # Target gather: compute target logit directly using dot product
                # Copy logits to shared memory for gather
                T.copy(logits, logits_shared)

                for i in T.Parallel(BLOCK_M):
                    row = bx * BLOCK_M + i
                    if row < T_block:
                        target = targets_chunk[row]
                        v_s = v_start[0]
                        v_o = v_off[0]
                        v_oe = v_off_end[0]
                        v_l = v_local[0]

                        in_rank = (target >= v_s) & (target < v_s + v_l)
                        in_chunk = in_rank & (target >= v_s + v_o + v_chunk_start) & (target < v_s + v_o + v_chunk_end)

                        if in_chunk:
                            local_idx = target - v_s - v_o
                            # Check if target is in this vocab chunk
                            if local_idx >= v_chunk_start and local_idx < v_chunk_end:
                                # Read logit from shared memory
                                local_target_logit[i] = local_target_logit[i] + logits_shared[i, local_idx - v_chunk_start]

            # Write outputs
            for i in T.Parallel(BLOCK_M):
                row = bx * BLOCK_M + i
                if row < T_block:
                    # Rescale sum_exp
                    rescale = T.exp(max_old[row] - local_max_new[i])
                    max_new[row] = local_max_new[i]
                    sum_exp_new[row] = sum_exp[row] * rescale + local_sum_exp[i]
                    target_logit_new[row] = target_logit[row] + local_target_logit[i]

    return fwd


@tilelang.jit(
    out_idx=[11, 12, 13],
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    compile_flags=["-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"],
)
def _tilelang_bwd_tile_kernel(
    T_block: int,
    H: int,
    V_tile: int,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
    has_bias: bool = True,
    inv_t: float = 1.0,
):
    """TileLang kernel for backward vocab tile.

    Computes:
        logits = hidden @ weight.T (+ bias, * inv_t)
        probs = exp(logits - logz)
        grad_logits = -grad * probs + grad * delta(target)
        grad_hidden = grad_logits @ weight
        grad_weight = grad_logits.T @ hidden
        grad_bias = sum(grad_logits, dim=0)

    Args:
        T_block: Sequence chunk size
        H: Hidden dimension
        V_tile: Vocab tile size
        dtype: Input dtype
        accum_dtype: Accumulation dtype
        has_bias: Whether bias is present
        inv_t: Inverse temperature scaling

    Returns:
        prim_func for the kernel
    """
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    @T.prim_func
    def bwd(
        hidden_chunk: T.Tensor([T_block, H], dtype),
        weight_chunk: T.Tensor([V_tile, H], dtype),
        bias_chunk: T.Tensor([V_tile], accum_dtype) if has_bias else T.Tensor([1], accum_dtype),
        targets_chunk: T.Tensor([T_block], T.int64),
        row_idx: T.Tensor([T_block], T.int64),
        grad_chunk: T.Tensor([T_block], accum_dtype),
        logz_chunk: T.Tensor([T_block], accum_dtype),
        v_start: T.Tensor([1], T.int32),
        v_off: T.Tensor([1], T.int32),
        v_off_end: T.Tensor([1], T.int32),
        v_local: T.Tensor([1], T.int32),
        grad_hidden_tile: T.Tensor([T_block, H], accum_dtype),
        grad_weight_tile: T.Tensor([V_tile, H], accum_dtype),
        grad_bias_tile: T.Tensor([V_tile], accum_dtype) if has_bias else T.Tensor([1], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(T_block, BLOCK_M), T.ceildiv(V_tile, BLOCK_N), threads=128) as (bx, by):
            # Shared memory for input tiles
            h_shared = T.alloc_shared([BLOCK_M, BLOCK_K], dtype)
            w_shared = T.alloc_shared([BLOCK_N, BLOCK_K], dtype)
            b_shared = T.alloc_shared([BLOCK_N], accum_dtype) if has_bias else T.alloc_shared([1], accum_dtype)

            # Fragments for logits and probs
            logits = T.alloc_fragment([BLOCK_M, BLOCK_N], accum_dtype)
            probs = T.alloc_fragment([BLOCK_M, BLOCK_N], accum_dtype)
            grad_logits = T.alloc_fragment([BLOCK_M, BLOCK_N], accum_dtype)

            # Fragments for gradient computation
            grad_logits_shared = T.alloc_shared([BLOCK_M, BLOCK_N], accum_dtype)
            grad_hidden_local = T.alloc_fragment([BLOCK_M, BLOCK_K], accum_dtype)
            grad_weight_local = T.alloc_fragment([BLOCK_N, BLOCK_K], accum_dtype)
            grad_bias_local = T.alloc_fragment([BLOCK_N], accum_dtype) if has_bias else T.alloc_fragment([1], accum_dtype)

            # Initialize accumulators
            T.clear(grad_hidden_local)
            T.clear(grad_weight_local)
            if has_bias:
                T.clear(grad_bias_local)

            # Compute logits for this vocab tile: h @ w.T
            T.clear(logits)
            for k in T.serial(T.ceildiv(H, BLOCK_K)):
                T.copy(hidden_chunk[bx * BLOCK_M, k * BLOCK_K], h_shared)
                T.copy(weight_chunk[by * BLOCK_N, k * BLOCK_K], w_shared)
                T.gemm(h_shared, w_shared, logits, transpose_B=True)

            # Add bias and apply inv_t
            if has_bias:
                T.copy(bias_chunk[by * BLOCK_N], b_shared)

            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                row = bx * BLOCK_M + i
                col = by * BLOCK_N + j
                if row < T_block and col < V_tile:
                    if has_bias:
                        logits[i, j] = logits[i, j] + b_shared[j]
                    if inv_t != 1.0:
                        logits[i, j] = logits[i, j] * inv_t

            # Compute probs = exp(logits - logz)
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                row = bx * BLOCK_M + i
                col = by * BLOCK_N + j
                if row < T_block and col < V_tile:
                    probs[i, j] = T.exp(logits[i, j] - logz_chunk[row])
                else:
                    probs[i, j] = 0.0

            # Compute grad_logits = -grad * probs
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                row = bx * BLOCK_M + i
                col = by * BLOCK_N + j
                if row < T_block and col < V_tile:
                    grad_logits[i, j] = -grad_chunk[row] * probs[i, j]
                else:
                    grad_logits[i, j] = 0.0

            # Target delta: add grad to grad_logits[row, local_idx] if target in chunk
            for i in T.Parallel(BLOCK_M):
                row = bx * BLOCK_M + i
                if row < T_block:
                    target = targets_chunk[row]
                    v_s = v_start[0]
                    v_o = v_off[0]
                    v_oe = v_off_end[0]
                    v_l = v_local[0]

                    in_rank = (target >= v_s) & (target < v_s + v_l)
                    in_chunk = in_rank & (target >= v_s + v_o) & (target < v_s + v_oe)

                    if in_chunk:
                        local_idx = target - v_s - v_o
                        # Check if target is in this vocab tile
                        if local_idx >= by * BLOCK_N and local_idx < (by + 1) * BLOCK_N:
                            grad_logits[i, local_idx - by * BLOCK_N] = grad_logits[i, local_idx - by * BLOCK_N] + grad_chunk[row]

            # Apply inv_t to grad_logits
            if inv_t != 1.0:
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row = bx * BLOCK_M + i
                    col = by * BLOCK_N + j
                    if row < T_block and col < V_tile:
                        grad_logits[i, j] = grad_logits[i, j] * inv_t

            # Copy grad_logits to shared memory for GEMM
            T.copy(grad_logits, grad_logits_shared)

            # Compute grad_hidden = grad_logits @ weight
            for k in T.serial(T.ceildiv(H, BLOCK_K)):
                T.copy(weight_chunk[by * BLOCK_N, k * BLOCK_K], w_shared)
                T.gemm(grad_logits_shared, w_shared, grad_hidden_local)

                # Accumulate to output
                for i, kk in T.Parallel(BLOCK_M, BLOCK_K):
                    row = bx * BLOCK_M + i
                    col = k * BLOCK_K + kk
                    if row < T_block and col < H:
                        T.atomic_add(grad_hidden_tile[row, col], grad_hidden_local[i, kk])

            # Compute grad_weight = grad_logits.T @ hidden
            for k in T.serial(T.ceildiv(H, BLOCK_K)):
                T.copy(hidden_chunk[bx * BLOCK_M, k * BLOCK_K], h_shared)
                T.gemm(grad_logits_shared, h_shared, grad_weight_local, transpose_A=True)

                # Accumulate to output
                for j, kk in T.Parallel(BLOCK_N, BLOCK_K):
                    row = by * BLOCK_N + j
                    col = k * BLOCK_K + kk
                    if row < V_tile and col < H:
                        T.atomic_add(grad_weight_tile[row, col], grad_weight_local[j, kk])

            # Compute grad_bias = sum(grad_logits, dim=0)
            if has_bias:
                T.reduce_sum(grad_logits, grad_bias_local, dim=0, clear=True)

                # Accumulate to output
                for j in T.Parallel(BLOCK_N):
                    col = by * BLOCK_N + j
                    if col < V_tile:
                        T.atomic_add(grad_bias_tile[col], grad_bias_local[j])

    return bwd


def tilelang_fwd_tile(
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
    """Forward vocab tile using TileLang kernel.

    Computes one vocab tile of the fused lm_head + log-prob forward pass.
    Returns updated online logsumexp accumulators and target logit.

    Args:
        hidden_chunk: [T, H] hidden states
        weight_chunk: [V_tile, H] weight matrix (vocab tile)
        bias_chunk: [V_tile] bias vector (vocab tile) or None
        targets_chunk: [T] target token ids
        row_idx: [T] row indices (for compatibility, not used in kernel)
        max_old: [T] running max from previous tiles
        sum_exp: [T] running sum of exp from previous tiles
        target_logit: [T] running target logit from previous tiles
        inv_t: inverse temperature scaling
        v_start: global vocab start index for this rank
        v_off: offset within rank's vocab range
        v_off_end: end offset within rank's vocab range
        v_local: total vocab size for this rank

    Returns:
        (max_new, sum_exp_new, target_logit_new): Updated accumulators
    """
    T_block, H = hidden_chunk.shape
    V_tile = weight_chunk.shape[0]
    has_bias = bias_chunk is not None and bias_chunk.numel() > 0

    # Allocate output tensors
    max_new = torch.empty_like(max_old)
    sum_exp_new = torch.empty_like(sum_exp)
    target_logit_new = torch.empty_like(target_logit)

    # Create scalar tensors for kernel arguments
    v_start_t = torch.tensor([v_start], dtype=torch.int32, device=hidden_chunk.device)
    v_off_t = torch.tensor([v_off], dtype=torch.int32, device=hidden_chunk.device)
    v_off_end_t = torch.tensor([v_off_end], dtype=torch.int32, device=hidden_chunk.device)
    v_local_t = torch.tensor([v_local], dtype=torch.int32, device=hidden_chunk.device)

    # Compile and run kernel
    kernel = _tilelang_fwd_tile_kernel(
        T_block,
        H,
        V_tile,
        dtype=hidden_chunk.dtype,
        accum_dtype=torch.float32,
        has_bias=has_bias,
        inv_t=inv_t,
    )

    # Pass dummy bias if None
    if not has_bias:
        bias_chunk = torch.zeros(1, device=hidden_chunk.device, dtype=torch.float32)

    max_new, sum_exp_new, target_logit_new = kernel(
        hidden_chunk,
        weight_chunk,
        bias_chunk,
        targets_chunk,
        row_idx,
        max_old,
        sum_exp,
        target_logit,
        v_start_t,
        v_off_t,
        v_off_end_t,
        v_local_t,
    )

    return max_new, sum_exp_new, target_logit_new


def tilelang_bwd_tile(
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
    """Backward vocab tile using TileLang kernel.

    Computes one vocab tile of the fused lm_head + log-prob backward pass.
    Returns local partial gradients for this tile only.

    Args:
        hidden_chunk: [T, H] hidden states
        weight_chunk: [V_tile, H] weight matrix (vocab tile)
        bias_chunk: [V_tile] bias vector (vocab tile) or None
        targets_chunk: [T] target token ids
        row_idx: [T] row indices (for compatibility, not used in kernel)
        grad_chunk: [T] upstream gradient
        logz_chunk: [T] log normalizer (logsumexp) for this tile
        inv_t: inverse temperature scaling
        v_start: global vocab start index for this rank
        v_off: offset within rank's vocab range
        v_off_end: end offset within rank's vocab range
        v_local: total vocab size for this rank

    Returns:
        (grad_hidden_tile, grad_weight_tile, grad_bias_tile): Local partial gradients
    """
    T_block, H = hidden_chunk.shape
    V_tile = weight_chunk.shape[0]
    has_bias = bias_chunk is not None and bias_chunk.numel() > 0

    # Allocate output tensors
    grad_hidden_tile = torch.zeros(T_block, H, device=hidden_chunk.device, dtype=torch.float32)
    grad_weight_tile = torch.zeros(V_tile, H, device=weight_chunk.device, dtype=torch.float32)
    grad_bias_tile = torch.zeros(V_tile, device=weight_chunk.device, dtype=torch.float32) if has_bias else None

    # Create scalar tensors for kernel arguments
    v_start_t = torch.tensor([v_start], dtype=torch.int32, device=hidden_chunk.device)
    v_off_t = torch.tensor([v_off], dtype=torch.int32, device=hidden_chunk.device)
    v_off_end_t = torch.tensor([v_off_end], dtype=torch.int32, device=hidden_chunk.device)
    v_local_t = torch.tensor([v_local], dtype=torch.int32, device=hidden_chunk.device)

    # Compile and run kernel
    kernel = _tilelang_bwd_tile_kernel(
        T_block,
        H,
        V_tile,
        dtype=hidden_chunk.dtype,
        accum_dtype=torch.float32,
        has_bias=has_bias,
        inv_t=inv_t,
    )

    # Pass dummy bias if None
    if not has_bias:
        bias_chunk = torch.zeros(1, device=hidden_chunk.device, dtype=torch.float32)
        grad_bias_tile = torch.zeros(1, device=hidden_chunk.device, dtype=torch.float32)

    grad_hidden_tile, grad_weight_tile, grad_bias_sum = kernel(
        hidden_chunk,
        weight_chunk,
        bias_chunk,
        targets_chunk,
        row_idx,
        grad_chunk,
        logz_chunk,
        v_start_t,
        v_off_t,
        v_off_end_t,
        v_local_t,
    )

    # The kernel returns the tile-local partials; hide them in the pre-allocated
    # (zeroed) buffers so downstream accumulation semantics match eager.
    hidden_out = grad_hidden_tile
    weight_out = grad_weight_tile
    bias_out = grad_bias_sum if has_bias else None

    return hidden_out, weight_out, bias_out
