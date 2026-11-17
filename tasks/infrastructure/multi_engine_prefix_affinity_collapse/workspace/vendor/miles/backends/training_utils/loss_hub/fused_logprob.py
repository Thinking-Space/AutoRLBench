"""Fused lm_head + log-prob -- one-phase PyTorch impl.

What "cross-entropy" means here
-------------------------------
There is no standalone CE *loss term* in the RL policy path. The "CE" this
module fuses is the **operator** ``log_softmax(logits).gather(target)`` -- i.e.
``log p(target)``, the log-probability of the target token. Mathematically this
is the log of the cross-entropy softmax denominator combined with the target
logit, so fusing "lm_head + CE" and fusing "lm_head + log-prob" are the same
thing.

The downstream consumer differs by loss type:

* **SFT** (``sft_loss_function``): consumes ``log p(target)`` directly as an
  explicit NLL loss: ``loss = -sum_of_sample_mean(log_probs)``. Here the fused
  log-prob *is* the (negated) CE loss term.
* **RL policy** (``policy_loss_function``): ``log p(target)`` enters the PPO
  ratio ``exp(log_probs - old_log_probs)`` and the clipped surrogate. CE is
  *implicit* in the ratio -- there is no separate CE term.

Two fusion boundaries
---------------------
The implementation maps to ``--fused-logprob``:

* **``fused_logprob``** -- :class:`FusedLogProbFunction` / :func:`fused_logprob`.
  Fuse lm_head + log-prob so the full ``[T, V]`` logits tensor never
  materializes in HBM. The per-token ``[T]`` log-prob (and optional ``[T]``
  entropy) *is* written back; the downstream PPO/SFT loss runs unchanged on
  that materialized ``[T]`` tensor (in the same autograd graph). Forward peak
  is tiny (just the ``[T]`` log-prob/log_z plus the per-chunk tile); the bulk
  of the work -- recomputing per-chunk logits and accumulating
  ``grad_hidden [T,H]`` + ``grad_weight [V_local,H]`` -- happens in
  :meth:`FusedLogProbFunction.backward`. The safe default.

Peak-memory:
    The heavy pass (backward) holds ``grad_input [T,H]`` + ``grad_weight
    [V_local,H]``. The full-pass peak is therefore dominated by those two
    tensors, NOT by ``[T,V]`` logits (which never materialize).

TP vocab-parallel adaptation (R11):
    ``weight`` is the local TP shard ``[V_local, H]`` of the (column-parallel)
    lm_head. Each rank owns vocab range ``[rank*V_local, (rank+1)*V_local)``.
    The online logsumexp runs over vocab chunks *within* a rank, then a single
    all-reduce over the TP group stitches the per-rank partials:

      * global ``max``: ``all_reduce(MAX)``
      * ``predicted_logit`` (target in this rank's range) + ``sum_exp``:
        packed into one tensor and reduced with ``all_reduce(SUM)``

    Backward recomputes per-chunk logits, reconstructs softmax probs from the
    saved ``log_z``, and accumulates ``grad_hidden``; a final ``all_reduce(SUM)``
    on ``grad_hidden`` stitches the partial sums from every TP rank (the hidden
    activations are replicated, so each rank's contribution must be summed).

This is a faithful port of Liger's ``_selective_logprob`` dual-chunking scheme
(``liger_kernel/chunked_loss/fused_linear_ppo.py``) with the TP collectives that
Liger omits (Liger assumes a single replicated lm_head). Pure PyTorch / eager --
no Triton, no liger-kernel dependency. Phase-2 will wrap this in
``torch.compile``; phase-3/4 swap the inner matmul+softmax for Triton/tilelang.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .fused_kernels import FUSED_KERNEL_STATE, fused_logprob_forward_tile, fused_logprob_backward_tile

# Default chunk sizes mirror Liger's _SELECTIVE_LOGPROB_* defaults. Tuned so a
# single ``[seq_chunk, vocab_chunk]`` tile fits comfortably in SRAM/L2 even for
# long sequences and large vocabularies; overridable via
# ``--fused-logprob-seq-chunk-size`` / ``--fused-logprob-vocab-chunk-size``.
_DEFAULT_SEQ_CHUNK_SIZE = 2048
_DEFAULT_VOCAB_CHUNK_SIZE = 4096


def set_fused_logprob_kernel(backend: str) -> None:
    """Set the kernel backend for fused-logprob tile operations."""
    FUSED_KERNEL_STATE.set(backend)


def reset_fused_kernel_state() -> None:
    """Clear kernel backend cache + re-enable the latch (tests use this)."""
    FUSED_KERNEL_STATE.reset()


def _tp_rank_size(group) -> tuple[int, int]:
    """Best-effort (rank, size) for a ProcessGroup (or None for single-process)."""
    if group is None:
        return 0, 1
    if not (dist.is_available() and dist.is_initialized()):
        return 0, 1
    return dist.get_rank(group), dist.get_world_size(group)


def _vocab_range(vocab_local: int, rank: int, world_size: int) -> tuple[int, int]:
    """Half-open ``[start, end)`` global vocab indices owned by ``rank``.

    Matches :class:`VocabUtility.vocab_range_from_per_partition_vocab_size`: the
    (column-parallel) lm_head shards vocab evenly, so rank ``r`` owns
    ``[r*V_local, (r+1)*V_local)``.
    """
    return rank * vocab_local, (rank + 1) * vocab_local


def _maybe_all_reduce(tensor: torch.Tensor, op, group) -> None:
    """In-place all-reduce; no-op when TP world size is 1 (single-process / tests)."""
    if group is None:
        return
    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_world_size(group) <= 1:
        return
    dist.all_reduce(tensor, op=op, group=group)


def fused_logprob_forward(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        bias: torch.Tensor | None,
        temperature: float,
        seq_chunk_size: int,
        vocab_chunk_size: int,
        group,
        *,
        vocab_range: tuple[int, int] | None = None,
        with_entropy: bool = False,
):
    """Stream over (seq x vocab) chunks; never materialize [T, V].

    Returns ``(log_probs[T], log_z[T])`` where ``log_probs = target_logit - log_z``.
    ``log_z`` is the per-token ``logsumexp`` over the *full global* vocab (stitched
    across TP ranks by all-reduce); it is what backward needs to rebuild softmax.

    When ``with_entropy=True`` the per-token entropy ``H = log_z - E_q[logit]`` is
    computed *in the same single vocab pass* as the log-prob, by carrying one extra
    online accumulator ``weighted_num = Σ_v p_v · logit_v`` through the same
    max-shift rescale used for ``sum_exp``. The ratio ``weighted_num / sum_exp``
    cancels the max-shift, yielding the true ``E_q[logit]`` without a second vocab
    scan . The third accumulator is packed with ``(target_logit, sum_exp)`` into one
    all-reduce(SUM), so the TP cost is unchanged vs. the no-entropy path. ``with_entropy``
    is a pure observed metric here; this function does not attach it to the autograd
    graph (the caller runs it under ``no_grad`` when entropy is requested).

    Args:
        hidden: ``[T, H]`` replicated across TP ranks.
        weight: ``[V_local, H]`` -- the TP-local shard of the lm_head.
        targets: ``[T]`` global token ids.
        bias: ``[V_local]`` or ``None`` -- TP-local shard of the lm_head bias.
        temperature: logits are divided by this before softmax.
        vocab_range: ``(start, end)`` global vocab indices owned by this rank.
            Computed from the group if omitted.
        with_entropy: also return the per-token entropy computed in this same pass
            (no extra vocab scan). When False the third element of the return is
            ``None``.
    """
    device = hidden.device
    n_rows = hidden.size(0)
    v_local = weight.size(0)
    inv_t = 1.0 / temperature if temperature != 0 else 1.0

    rank, world = _tp_rank_size(group)
    if vocab_range is None:
        vocab_range = _vocab_range(v_local, rank, world)
    v_start, _v_end = vocab_range

    logprobs = torch.empty((n_rows,), device=device, dtype=torch.float32)
    log_z = torch.empty((n_rows,), device=device, dtype=torch.float32)
    entropy = torch.empty((n_rows,), device=device, dtype=torch.float32) if with_entropy else None

    for seq_start in range(0, n_rows, seq_chunk_size):
        seq_end = min(seq_start + seq_chunk_size, n_rows)
        n_chunk = seq_end - seq_start
        hidden_chunk = hidden[seq_start:seq_end]
        targets_chunk = targets[seq_start:seq_end]

        # Online logsumexp over vocab chunks *within* this rank. The global
        # max/sum across TP ranks are stitched by all-reduce AFTER the local
        # stream, so the in-chunk accumulators are rank-local partials.
        max_old = torch.full((n_chunk,), float("-inf"), device=device, dtype=torch.float32)
        sum_exp = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
        target_logit = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
        # E_q[logit] numerator, accumulated against the SAME max_new as sum_exp
        # so weighted_num / sum_exp cancels the max-shift. Only allocated when we
        # actually need entropy (keeps the no-entropy path byte-for-byte unchanged).
        weighted_num = (
            torch.zeros((n_chunk,), device=device, dtype=torch.float32) if with_entropy else None
        )
        row_idx = torch.arange(n_chunk, device=device)

        for v_off in range(0, v_local, vocab_chunk_size):
            v_off_end = min(v_off + vocab_chunk_size, v_local)
            max_old, sum_exp, target_logit, weighted_num = fused_logprob_forward_tile(
                hidden_chunk,
                weight[v_off:v_off_end],
                None if bias is None else bias[v_off:v_off_end],
                targets_chunk,
                row_idx,
                max_old,
                sum_exp,
                target_logit,
                inv_t,
                v_start,
                v_off,
                v_off_end,
                v_local,
                weighted_num=weighted_num,
            )

        # Stitch across TP ranks: one MAX for the global max, one SUM for the
        # packed (target_logit, sum_exp[, weighted_num]) tuple. target_logit is
        # nonzero only on the rank that owns the target, so summing recovers the
        # single contribution. ``sum_exp`` / ``weighted_num`` were accumulated
        # against this rank's *local* max, so after the global MAX arrives we
        # must rescale by ``exp(local_max - global_max)`` before summing the
        # partials -- the same rescale applies to weighted_num since it shares
        # max_new with sum_exp.
        #
        # NOTE: ``dist.all_reduce`` is *in-place*, so snapshot the local max
        # (clone) before the MAX reduce -- otherwise ``max_old`` is overwritten
        # in place with the global max and the rescale below degenerates to 1.
        local_max = max_old.clone()
        _maybe_all_reduce(max_old, dist.ReduceOp.MAX, group)
        rescale = torch.exp(local_max - max_old)
        sum_exp = sum_exp * rescale
        if weighted_num is not None:
            weighted_num = weighted_num * rescale
            packed = torch.stack([target_logit, sum_exp, weighted_num], dim=0)  # [3, n_chunk]
        else:
            packed = torch.stack([target_logit, sum_exp], dim=0)  # [2, n_chunk]
        _maybe_all_reduce(packed, dist.ReduceOp.SUM, group)
        target_logit, sum_exp = packed[0], packed[1]
        if weighted_num is not None:
            weighted_num = packed[2]

        log_z_chunk = max_old + torch.log(sum_exp)
        log_z[seq_start:seq_end] = log_z_chunk
        logprobs[seq_start:seq_end] = target_logit - log_z_chunk
        if weighted_num is not None:
            # H = log_z - E_q[logit]; E_q[logit] = weighted_num / sum_exp (max-shift
            # cancels in the ratio). sum_exp > 0 by construction (every token has at
            # least one finite logit across the full vocab).
            entropy[seq_start:seq_end] = log_z_chunk - weighted_num / sum_exp

    return logprobs, log_z, entropy


def fused_logprob_backward(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        bias: torch.Tensor | None,
        log_z: torch.Tensor,
        grad_logprobs: torch.Tensor,
        temperature: float,
        seq_chunk_size: int,
        vocab_chunk_size: int,
        group,
        *,
        vocab_range: tuple[int, int] | None = None,
):
    """Backward for the fused log-prob.

    Recomputes per-chunk logits (never stored), rebuilds softmax probs from
    ``log_z`` (full-vocab logsumexp, already stitched across TP), and accumulates
    grads. The Jacobian of ``logprob_t = logit_t - logsumexp`` is::

        d logprob_t / d logit_v = delta_{v,t} - softmax(logit)_v

    so ``grad_logit_v = -grad_logprob_t * p_{t,v} + grad_logprob_t * delta_{v,t}``
    (per row t). We stream the same (seq x vocab) chunks as forward.

    Final ``all_reduce(SUM)`` on ``grad_hidden`` stitches the partials from every
    TP rank (hidden is replicated; every rank computes a partial contribution to
    the same grad).
    """
    device = hidden.device
    inv_t = 1.0 / temperature if temperature != 0 else 1.0
    n_rows, hidden_dim = hidden.shape
    v_local = weight.size(0)
    has_bias = bias is not None
    rank, world = _tp_rank_size(group)
    if vocab_range is None:
        vocab_range = _vocab_range(v_local, rank, world)
    v_start, _v_end = vocab_range

    grad_hidden = torch.zeros(hidden.shape, device=device, dtype=torch.float32)
    grad_weight = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
    grad_bias = torch.zeros((v_local,), device=weight.device, dtype=torch.float32) if has_bias else None
    grad_logprobs = grad_logprobs.to(torch.float32)

    for seq_start in range(0, n_rows, seq_chunk_size):
        seq_end = min(seq_start + seq_chunk_size, n_rows)
        hidden_chunk = hidden[seq_start:seq_end]
        targets_chunk = targets[seq_start:seq_end]
        grad_chunk = grad_logprobs[seq_start:seq_end]
        logz_chunk = log_z[seq_start:seq_end]
        row_idx = torch.arange(seq_end - seq_start, device=device)

        for v_off in range(0, v_local, vocab_chunk_size):
            v_off_end = min(v_off + vocab_chunk_size, v_local)
            grad_hidden_tile, grad_weight_tile, grad_bias_tile = fused_logprob_backward_tile(
                hidden_chunk,
                weight[v_off:v_off_end],
                None if not has_bias else bias[v_off:v_off_end],
                targets_chunk,
                row_idx,
                grad_chunk,
                logz_chunk,
                inv_t,
                v_start,
                v_off,
                v_off_end,
                v_local,
            )
            grad_hidden[seq_start:seq_end].add_(grad_hidden_tile)
            grad_weight[v_off:v_off_end].add_(grad_weight_tile)
            if has_bias:
                grad_bias[v_off:v_off_end].add_(grad_bias_tile)

    # hidden is replicated across TP ranks; sum the per-rank partial grad_hidden.
    _maybe_all_reduce(grad_hidden, dist.ReduceOp.SUM, group)

    return grad_hidden, grad_weight, grad_bias


class FusedLogProbFunction(torch.autograd.Function):
    """Fused lm_head + selected-token log-prob (``--fused-logprob``).

    Forward streams (seq x vocab) chunks so the full ``[T, V]`` logits never hit
    HBM; backward recomputes per chunk. Saves only ``hidden``, ``weight``,
    ``targets``, ``bias``, ``log_z`` (and ``vocab_range``/temperature as plain
    attrs) -- a few ``[T]`` / ``[T,H]`` tensors, no ``[T,V]``. The ``[T]``
    log-prob is returned to the caller (materialized); the downstream PPO/SFT
    loss reads it in the same autograd graph.

    The forward *always* returns ``(log_probs, entropy)``. ``entropy`` is
    ``None`` when ``with_entropy=False``; otherwise it is the per-token entropy
    computed in the *same single vocab pass* as the log-prob (see
    :func:`_fused_logprob_forward`'s ``with_entropy`` arg) under ``torch.no_grad``
    -- a pure observed metric, never part of the autograd graph. Fused entropy
    backward is not implemented (the entropy term's gradient would need the
    full-vocab logits we deliberately avoid materializing), so the caller guards
    against ``entropy_coef != 0`` and the entropy output's grad is always ignored
    in :meth:`backward`.
    """

    @staticmethod
    def forward(
            ctx,
            hidden,
            weight,
            targets,
            bias,
            temperature,
            seq_chunk_size,
            vocab_chunk_size,
            group,
            with_entropy,
    ):
        # Single vocab pass: log-prob (+ entropy in the same pass when requested).
        # Entropy is an observed metric whose gradient is discarded in backward
        # (``del grad_entropy``); it is NOT isolated via no_grad here -- wrapping
        # the whole forward in no_grad would also detach ``logprobs`` and break
        # backward. entropy shares this pass's logits but its grad is simply
        # ignored upstream, and entropy is not saved_for_backward, so it adds no
        # backward recompute or memory.
        logprobs, log_z, entropy = fused_logprob_forward(
            hidden,
            weight,
            targets,
            bias,
            temperature,
            seq_chunk_size,
            vocab_chunk_size,
            group,
            with_entropy=with_entropy,
        )

        # Normalize None bias to a 0-length tensor for save_for_backward round-trip.
        if bias is None:
            bias = hidden.new_empty((0,))
        ctx.save_for_backward(hidden, weight, targets, bias, log_z)
        ctx.has_bias = bias.numel() > 0
        ctx.temperature = temperature
        ctx.seq_chunk_size = seq_chunk_size
        ctx.vocab_chunk_size = vocab_chunk_size
        ctx.group = group
        # vocab_range is recomputed in backward from group + weight size; no need to save.
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs, grad_entropy):
        # ``entropy`` is computed under no_grad (or None); its grad is unused.
        # The caller rejects ``entropy_coef != 0`` at the entry point, so no
        # gradient ever flows through the entropy output.
        del grad_entropy
        hidden, weight, targets, bias, log_z = ctx.saved_tensors
        group = ctx.group
        rank, world = _tp_rank_size(group)
        v_local = weight.size(0)
        vocab_range = _vocab_range(v_local, rank, world)

        grad_hidden, grad_weight, grad_bias = fused_logprob_backward(
            hidden=hidden,
            weight=weight,
            targets=targets,
            bias=bias if ctx.has_bias else None,
            log_z=log_z,
            grad_logprobs=grad_logprobs,
            temperature=ctx.temperature,
            seq_chunk_size=ctx.seq_chunk_size,
            vocab_chunk_size=ctx.vocab_chunk_size,
            group=group,
            vocab_range=vocab_range,
        )

        return (
            grad_hidden.to(hidden.dtype),
            grad_weight.to(weight.dtype),
            None,  # targets
            grad_bias.to(bias.dtype) if ctx.has_bias else None,
            None,  # temperature
            None,  # seq_chunk_size
            None,  # vocab_chunk_size
            None,  # group
            None,  # with_entropy
        )


def fused_linear_log_probs_and_entropy(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        tokens: torch.Tensor,
        group,
        *,
        bias: torch.Tensor | None = None,
        temperature: float = 1.0,
        with_entropy: bool = False,
        entropy_requires_grad: bool = True,
        seq_chunk_size: int = _DEFAULT_SEQ_CHUNK_SIZE,
        vocab_chunk_size: int = _DEFAULT_VOCAB_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop-in for :func:`calculate_log_probs_and_entropy` on the fused path.

    Mirrors the (``log_prob, entropy``) return shape of the non-fused helper so
    the fused entry points can branch on ``args.fused_logprob`` with no other
    changes. ``with_entropy=False`` skips the extra vocab-streaming entropy pass
    (and its TP all-reduce) entirely and returns ``entropy=None``, matching the
    non-fused helper's contract.

    Equivalent to::

        logits = (hidden @ weight.T + bias) / temperature   # full [T, V]
        log_probs = log_softmax(logits, dim=-1).gather(-1, tokens)

    but the ``[T, V]`` logits never materialize -- peak temp memory is
    ``seq_chunk_size * vocab_chunk_size``.

    Args:
        hidden: ``[T, H]`` float (fp32 / bf16), replicated across TP ranks.
        weight: ``[V_local, H]`` -- TP-local shard of the (column-parallel)
            lm_head. ``weight`` is the local shard even when TP=1 (then
            ``V_local == V_global``).
        tokens: ``[T]`` global token ids (int64).
        group: TP process group (or None for single-process).
        bias: ``[V_local]`` optional TP-local bias shard.
        temperature: logits are divided by this before softmax.
        with_entropy: also return per-token entropy ``[T]`` (observed metric;
            never differentiated on the fused path).
        entropy_requires_grad: the fused path cannot differentiate entropy
            (its backward would need the full-vocab logits we avoid
            materializing), so this must be ``False`` whenever entropy is
            requested -- i.e. ``entropy_coef == 0``. ``True`` *and*
            ``with_entropy=True`` raises; use the non-fused path or
            ``--entropy-coef 0``. Ignored when ``with_entropy=False``.
        seq_chunk_size / vocab_chunk_size: streaming tile sizes.

    Returns:
        ``(log_probs, entropy)`` -- ``entropy`` is ``None`` when
        ``with_entropy=False``.
    """
    if with_entropy and entropy_requires_grad:
        raise ValueError(
            "The fused loss path does not support entropy_coef != 0 (entropy "
            "must contribute to the loss gradient, but fused entropy backward "
            "is not implemented). Use the non-fused path or set --entropy-coef 0."
        )
    if hidden.dim() != 2:
        raise ValueError(f"hidden must be [T, H], got {tuple(hidden.shape)}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be [V_local, H], got {tuple(weight.shape)}")
    if hidden.size(-1) != weight.size(-1):
        raise ValueError(
            f"hidden/weight hidden-dim mismatch: {hidden.size(-1)} vs {weight.size(-1)}"
        )
    targets = tokens.to(torch.int64)
    if targets.dim() != 1 or targets.size(0) != hidden.size(0):
        raise ValueError(f"targets must be [T] matching hidden, got {tuple(targets.shape)}")
    if seq_chunk_size <= 0:
        seq_chunk_size = _DEFAULT_SEQ_CHUNK_SIZE
    if vocab_chunk_size <= 0:
        vocab_chunk_size = _DEFAULT_VOCAB_CHUNK_SIZE

    log_prob, entropy = FusedLogProbFunction.apply(
        hidden,
        weight,
        targets,
        bias,
        temperature,
        seq_chunk_size,
        vocab_chunk_size,
        group,
        with_entropy,
    )
    return log_prob, entropy
