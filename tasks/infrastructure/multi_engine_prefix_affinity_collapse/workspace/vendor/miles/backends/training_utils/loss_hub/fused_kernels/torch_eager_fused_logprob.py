"""Pure-PyTorch eager backend for fused logprob forward/backward tile operations.

This is the ``torch_eager`` backend: the pure-tensor tile math with no
``torch.compile`` wrapping (and no custom kernel). It is the safe baseline --
identical numerics to ``torch_compile`` (which wraps the same math in
``torch.compile(dynamic=True)``) but with no JIT, so no compile cost and no
graph-capture failure modes.

Both ``torch_eager`` and ``torch_compile`` share the *same* tile implementation
body (``fwd_tile_impl`` / ``bwd_tile_impl`` here); ``torch_compile`` lives in
its own module (:mod:`...fused_kernels.torch_compile_fused_logprob`) and wraps these
in ``torch.compile``. Keeping them as sibling modules means the dispatcher in
``fused_logprob.py`` treats torch the same way it treats triton / tilelang /
cute_dsl -- a lazily-loaded backend exporting ``fwd_tile_impl`` /
``bwd_tile_impl`` matching the eager signature documented in
:mod:`...fused_kernels.__init__`.

The tile math computes rank-local partials only; TP collectives
(all-reduce MAX/SUM) are handled by the Python host in ``fused_logprob.py``.
"""

from __future__ import annotations

import torch


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
    *,
    weighted_num: torch.Tensor | None = None,
):
    """One forward vocab tile (pure tensor math, compile-friendly).

    Functional (returns new accumulators rather than mutating in place) so it is
    safe under ``torch.compile``. The caller threads the running online-logsumexp
    accumulators (``max_old`` / ``sum_exp`` / ``target_logit``) through the vocab
    loop; the TP cross-rank stitch is done by the caller *after* this loop.

    When ``weighted_num`` is provided (entropy requested), this also folds the
    per-tile ``Σ_v p_v · logit_v`` contribution into it using the same online
    max-shift rescale as ``sum_exp``: the running ``weighted_num`` is rescaled by
    ``exp(max_old - max_new)`` and the new chunk adds ``Σ (exp(logit - max_new) · logit)``.
    The caller divides ``weighted_num / sum_exp`` (max-shift cancels) to get
    ``E_q[logit]`` for the entropy, with no extra vocab scan.

    Returns the updated ``(max_old, sum_exp, target_logit, weighted_num)``;
    ``weighted_num`` is the unchanged input when it was ``None``.
    """
    logits_chunk = (hidden_chunk @ weight_chunk.to(hidden_chunk.dtype).t()).float()
    if bias_chunk is not None:
        logits_chunk = logits_chunk + bias_chunk.to(torch.float32)
    if inv_t != 1.0:
        logits_chunk = logits_chunk * inv_t

    chunk_max = logits_chunk.amax(dim=-1)
    max_new = torch.maximum(max_old, chunk_max)
    rescale = torch.exp(max_old - max_new)
    chunk_exp = torch.exp(logits_chunk - max_new.unsqueeze(-1))

    sum_exp = sum_exp * rescale + chunk_exp.sum(dim=-1)

    if weighted_num is not None:
        # E_q[logit] numerator: rescale the running sum (same max_new as sum_exp)
        # and add this chunk's Σ p_v · logit_v (p_v = exp(logit_v - max_new)).
        weighted_num = weighted_num * rescale + (chunk_exp * logits_chunk).sum(dim=-1)

    # Target token -> global id; is it in this rank's range and this chunk?
    global_token = targets_chunk  # global ids
    in_rank = (global_token >= v_start) & (global_token < v_start + v_local)
    local_token = torch.clamp(global_token - v_start, 0, v_local - 1)
    in_chunk = in_rank & (global_token >= v_start + v_off) & (global_token < v_start + v_off_end)
    local_idx = torch.clamp(local_token - v_off, 0, v_off_end - v_off - 1)
    target_logit = target_logit + logits_chunk[row_idx, local_idx] * in_chunk

    return max_new, sum_exp, target_logit, weighted_num


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
    """One backward vocab tile (pure tensor math, compile-friendly).

    Returns the *local* partials for this tile only -- ``(grad_hidden_tile
    [seq_chunk, H], grad_weight_tile [tile_vocab, H], grad_bias_tile
    [tile_vocab])`` -- so the caller can accumulate in-place into the full-size
    ``grad_hidden``/``grad_weight``/``grad_bias`` buffers (keeping those big
    buffers out of the compile scope and avoiding a fresh full-size copy per
    tile). The TP ``all_reduce(SUM)`` on ``grad_hidden`` is done by the caller
    after the loop.
    """
    logits_chunk = (hidden_chunk @ weight_chunk.to(hidden_chunk.dtype).t()).float()
    if bias_chunk is not None:
        logits_chunk = logits_chunk + bias_chunk.to(torch.float32)
    if inv_t != 1.0:
        logits_chunk = logits_chunk * inv_t

    probs = torch.exp(logits_chunk - logz_chunk.unsqueeze(-1))
    # grad_logits = -grad * probs + grad * delta(target)  (per row)
    grad_logits = (-grad_chunk).unsqueeze(-1) * probs
    global_token = targets_chunk
    in_rank = (global_token >= v_start) & (global_token < v_start + v_local)
    in_chunk = in_rank & (global_token >= v_start + v_off) & (global_token < v_start + v_off_end)
    local_idx = torch.clamp(global_token - v_start, 0, v_local - 1)
    local_idx = torch.clamp(local_idx - v_off, 0, v_off_end - v_off - 1)
    grad_logits = grad_logits.scatter(
        1, local_idx.unsqueeze(-1), grad_logits.gather(1, local_idx.unsqueeze(-1)) + grad_chunk.unsqueeze(-1) * in_chunk.unsqueeze(-1)
    )
    if inv_t != 1.0:
        grad_logits = grad_logits * inv_t

    grad_hidden_tile = grad_logits @ weight_chunk.float()
    grad_weight_tile = grad_logits.t() @ hidden_chunk.float()
    grad_bias_tile = grad_logits.sum(dim=0) if bias_chunk is not None else None
    return grad_hidden_tile, grad_weight_tile, grad_bias_tile
