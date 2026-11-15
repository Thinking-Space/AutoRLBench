"""Triton backend for fused logprob forward/backward tile operations.

This module provides Triton kernel implementations of the fused logprob tile
operations, matching the semantics of the PyTorch eager implementations in
fused_logprob.py. The fused_kernels compute rank-local partials only; TP collectives
(all-reduce MAX/SUM) are handled by the Python host.

Forward tile: computes logits = hidden @ weight.T (+ bias, * inv_t), performs the
online logsumexp update (max_new, rescale, sum_exp) by looping over vocab
column tiles inside each row-block program, and gathers the target logit.

Backward tile: recomputes logits, computes softmax probabilities against the final
log-partition ``logz``, applies the target delta, then chains to
grad_hidden/grad_weight/grad_bias.

Both fused_kernels use a 1-D row-block grid; the vocab dimension is looped inside each
program so the online (forward) logsumexp and the (backward) accumulation are done
sequentially per row block. This mirrors the single-program-per-row FlashAttention /
online-softmax pattern.

Recompilation hazard
--------------------
``seq_chunk`` (the per-tile row count = each sample's response length) and
``vocab_chunk`` (the per-tile column count, the last vocab tile is a remainder)
are **runtime** parameters, NOT ``tl.constexpr``. Making them constexpr bakes the
shape into the kernel hash, so every sample (different response length) and every
remainder vocab tile triggers a fresh ~4s ``ptxas`` compile -- 512 samples/rank
meant the whole run stalled in the Triton compiler. Only the fixed block sizes
``BLOCK_M`` / ``BLOCK_N`` / ``BLOCK_K`` are constexpr; row/column bounds are
enforced with masks so a single compiled kernel serves every tile shape.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_tile_kernel(
        hidden_ptr,
        weight_ptr,
        bias_ptr,
        targets_ptr,
        max_old_ptr,
        sum_exp_ptr,
        target_logit_ptr,
        max_new_ptr,
        sum_exp_out_ptr,
        target_logit_out_ptr,
        seq_chunk,
        vocab_chunk,
        hidden_dim,
        inv_t: tl.constexpr,
        v_start,
        v_off,
        v_off_end,
        v_local: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    """Forward tile kernel: online logsumexp + target gather.

    Grid: (cdiv(seq_chunk, BLOCK_M),). Each program handles one row block and
    loops over the vocab column tiles sequentially, keeping the online
    logsumexp update correct (the same dependency that eager computes with a
    full ``[T, V]`` tile). ``seq_chunk`` / ``vocab_chunk`` / ``v_start`` /
    ``v_off`` / ``v_off_end`` are runtime ints so the kernel is compiled once
    per (block size, dtype) and reused across all tile shapes (every sample's
    response length, every vocab tile offset, every remainder). Only the fixed
    block sizes and ``v_local`` (constant within a rank) stay constexpr.
    """
    pid_m = tl.program_id(0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)

    row_mask = rm < seq_chunk
    col_valid = rk[None, :] < hidden_dim

    # Load the running accumulators for these rows.
    max_old = tl.load(max_old_ptr + rm, mask=row_mask, other=float("-inf"))
    sum_exp = tl.load(sum_exp_ptr + rm, mask=row_mask, other=0.0)
    target_logit = tl.load(target_logit_ptr + rm, mask=row_mask, other=0.0)
    targets = tl.load(targets_ptr + rm, mask=row_mask, other=0)

    # hidden [BLOCK_M, BLOCK_K] (only rows-load; reduced over vocab via dot).
    hidden = tl.load(
        hidden_ptr + rm[:, None] * hidden_dim + rk[None, :],
        mask=row_mask[:, None] & col_valid,
        other=0.0,
    )

    num_n = tl.cdiv(vocab_chunk, BLOCK_N)
    for nb in range(num_n):
        rn = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        lane_n = tl.arange(0, BLOCK_N)
        col_mask = rn < vocab_chunk

        # weight [BLOCK_N, BLOCK_K]
        weight = tl.load(
            weight_ptr + rn[:, None] * hidden_dim + rk[None, :],
            mask=col_mask[:, None] & col_valid,
            other=0.0,
        )
        # logits [BLOCK_M, BLOCK_N] = hidden @ weight.T
        logits = tl.dot(hidden, tl.trans(weight))
        if has_bias:
            bias = tl.load(bias_ptr + rn, mask=col_mask, other=0.0)
            logits = logits + bias[None, :]
        if inv_t != 1.0:
            logits = logits * inv_t

        # Mask out-of-range columns so they never win the max / contribute to
        # sum_exp (they loaded as 0 logits -> exp(0)=1 would pollute the sum).
        logits = tl.where(col_mask[None, :], logits, float("-inf"))

        chunk_max = tl.max(logits, axis=1)
        max_new = tl.maximum(max_old, chunk_max)
        rescale = tl.exp(max_old - max_new)
        chunk_exp = tl.exp(logits - max_new[:, None])
        sum_exp = sum_exp * rescale + tl.sum(chunk_exp, axis=1)
        max_old = max_new

        # Target gather: a target token belongs to exactly one vocab column tile.
        local_idx = targets - (v_start + v_off)
        in_this_n = (
                (targets >= v_start + v_off)
                & (targets < v_start + v_off_end)
                & (local_idx >= nb * BLOCK_N)
                & (local_idx < (nb + 1) * BLOCK_N)
        )
        li = tl.minimum(tl.maximum(local_idx - nb * BLOCK_N, 0), BLOCK_N - 1)
        gathered = tl.sum(tl.where(lane_n[None, :] == li[:, None], logits, 0.0), axis=1)
        target_logit = target_logit + gathered * in_this_n

    tl.store(max_new_ptr + rm, max_old, mask=row_mask)
    tl.store(sum_exp_out_ptr + rm, sum_exp, mask=row_mask)
    tl.store(target_logit_out_ptr + rm, target_logit, mask=row_mask)


@triton.jit
def _bwd_tile_kernel(
        hidden_ptr,
        weight_ptr,
        bias_ptr,
        targets_ptr,
        grad_ptr,
        logz_ptr,
        grad_hidden_ptr,
        grad_weight_ptr,
        grad_bias_ptr,
        seq_chunk,
        vocab_chunk,
        hidden_dim,
        inv_t: tl.constexpr,
        v_start,
        v_off,
        v_off_end,
        v_local: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    """Backward tile kernel: recompute logits, chain grads.

    Grid: (cdiv(seq_chunk, BLOCK_M),). Each program owns one row block,
    accumulates ``grad_hidden`` in registers over all vocab tiles, and
    atomically accumulates ``grad_weight`` / ``grad_bias`` (which fold over
    row blocks). ``seq_chunk`` / ``vocab_chunk`` / ``v_start`` / ``v_off`` /
    ``v_off_end`` are runtime ints (see :func:`_fwd_tile_kernel`); out-of-range
    columns are masked to ``-inf`` so they contribute zero probability and
    zero gradient.
    """
    pid_m = tl.program_id(0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)

    row_mask = rm < seq_chunk
    col_valid = rk[None, :] < hidden_dim

    hidden = tl.load(
        hidden_ptr + rm[:, None] * hidden_dim + rk[None, :],
        mask=row_mask[:, None] & col_valid,
        other=0.0,
    )
    grad = tl.load(grad_ptr + rm, mask=row_mask, other=0.0)
    logz = tl.load(logz_ptr + rm, mask=row_mask, other=0.0)
    targets = tl.load(targets_ptr + rm, mask=row_mask, other=0)

    local_idx = targets - (v_start + v_off)

    grad_hidden_acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    num_n = tl.cdiv(vocab_chunk, BLOCK_N)
    for nb in range(num_n):
        rn = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        lane_n = tl.arange(0, BLOCK_N)
        col_mask = rn < vocab_chunk

        weight = tl.load(
            weight_ptr + rn[:, None] * hidden_dim + rk[None, :],
            mask=col_mask[:, None] & col_valid,
            other=0.0,
        )
        logits = tl.dot(hidden, tl.trans(weight))
        if has_bias:
            bias = tl.load(bias_ptr + rn, mask=col_mask, other=0.0)
            logits = logits + bias[None, :]
        if inv_t != 1.0:
            logits = logits * inv_t

        # Out-of-range columns: zero probability, zero gradient.
        logits = tl.where(col_mask[None, :], logits, float("-inf"))

        probs = tl.exp(logits - logz[:, None])
        grad_logits = (-grad)[:, None] * probs

        # Target delta: grad_logits[row, target] += grad.
        in_this_n = (
                (targets >= v_start + v_off)
                & (targets < v_start + v_off_end)
                & (local_idx >= nb * BLOCK_N)
                & (local_idx < (nb + 1) * BLOCK_N)
        )
        li = tl.minimum(tl.maximum(local_idx - nb * BLOCK_N, 0), BLOCK_N - 1)
        add_mask = (lane_n[None, :] == li[:, None]) & in_this_n[:, None]
        grad_logits = tl.where(add_mask, grad_logits + grad[:, None], grad_logits)

        if inv_t != 1.0:
            grad_logits = grad_logits * inv_t

        grad_hidden_acc += tl.dot(grad_logits, weight)

        grad_weight = tl.dot(tl.trans(grad_logits), hidden)
        tl.atomic_add(
            grad_weight_ptr + rn[:, None] * hidden_dim + rk[None, :],
            grad_weight,
            mask=col_mask[:, None] & col_valid,
        )
        if has_bias:
            grad_bias = tl.sum(grad_logits, axis=0)
            tl.atomic_add(grad_bias_ptr + rn, grad_bias, mask=col_mask)

    tl.store(
        grad_hidden_ptr + rm[:, None] * hidden_dim + rk[None, :],
        grad_hidden_acc,
        mask=row_mask[:, None] & col_valid,
    )


def fwd_tile_impl(
        hidden_chunk: torch.Tensor,
        weight_chunk: torch.Tensor,
        bias_chunk: torch.Tensor | None,
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
    """Triton forward tile implementation (matches eager signature)."""
    seq_chunk, hidden_dim = hidden_chunk.shape
    vocab_chunk = weight_chunk.shape[0]
    device = hidden_chunk.device

    hidden_chunk = hidden_chunk.float()
    weight_chunk = weight_chunk.float()
    if bias_chunk is not None:
        bias_chunk = bias_chunk.float()

    max_new = torch.empty_like(max_old)
    sum_exp_out = torch.empty_like(sum_exp)
    target_logit_out = torch.empty_like(target_logit)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = triton.next_power_of_2(hidden_dim)

    grid = (triton.cdiv(seq_chunk, BLOCK_M),)
    _fwd_tile_kernel[grid](
        hidden_chunk,
        weight_chunk,
        bias_chunk if bias_chunk is not None else hidden_chunk.new_empty(0),
        targets_chunk,
        max_old,
        sum_exp,
        target_logit,
        max_new,
        sum_exp_out,
        target_logit_out,
        seq_chunk=seq_chunk,
        vocab_chunk=vocab_chunk,
        hidden_dim=hidden_dim,
        inv_t=inv_t,
        v_start=v_start,
        v_off=v_off,
        v_off_end=v_off_end,
        v_local=v_local,
        has_bias=bias_chunk is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return max_new, sum_exp_out, target_logit_out


def bwd_tile_impl(
        hidden_chunk: torch.Tensor,
        weight_chunk: torch.Tensor,
        bias_chunk: torch.Tensor | None,
        targets_chunk: torch.Tensor,
        row_idx: torch.Tensor,
        grad_chunk: torch.Tensor,
        logz_chunk: torch.Tensor,
        inv_t: float,
        v_start: int,
        v_off: int,
        v_off_end: int,
        v_local: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Triton backward tile implementation (matches eager signature)."""
    seq_chunk, hidden_dim = hidden_chunk.shape
    vocab_chunk = weight_chunk.shape[0]

    hidden_chunk = hidden_chunk.float()
    weight_chunk = weight_chunk.float()
    if bias_chunk is not None:
        bias_chunk = bias_chunk.float()
    grad_chunk = grad_chunk.float()
    logz_chunk = logz_chunk.float()

    device = hidden_chunk.device
    grad_hidden = torch.zeros((seq_chunk, hidden_dim), device=device, dtype=torch.float32)
    grad_weight = torch.zeros((vocab_chunk, hidden_dim), device=device, dtype=torch.float32)
    grad_bias = (
        torch.zeros((vocab_chunk,), device=device, dtype=torch.float32)
        if bias_chunk is not None
        else None
    )

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = triton.next_power_of_2(hidden_dim)

    grid = (triton.cdiv(seq_chunk, BLOCK_M),)
    _bwd_tile_kernel[grid](
        hidden_chunk,
        weight_chunk,
        bias_chunk if bias_chunk is not None else hidden_chunk.new_empty(0),
        targets_chunk,
        grad_chunk,
        logz_chunk,
        grad_hidden,
        grad_weight,
        grad_bias if grad_bias is not None else hidden_chunk.new_empty(0),
        seq_chunk=seq_chunk,
        vocab_chunk=vocab_chunk,
        hidden_dim=hidden_dim,
        inv_t=inv_t,
        v_start=v_start,
        v_off=v_off,
        v_off_end=v_off_end,
        v_local=v_local,
        has_bias=bias_chunk is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return grad_hidden, grad_weight, grad_bias
