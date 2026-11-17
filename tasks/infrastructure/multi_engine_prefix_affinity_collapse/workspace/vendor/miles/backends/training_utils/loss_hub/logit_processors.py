from argparse import Namespace
from collections.abc import Iterator

import torch

from miles.backends.training_utils.cp_utils import allgather_cp_redistribute, get_logits_and_tokens_offset_with_cp
from miles.backends.training_utils.loss_hub.fused_logprob import fused_linear_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.timer import Timer

from megatron.core import tensor_parallel


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    qkv_format = args.qkv_format

    if not args.true_on_policy_mode:
        # FSDP hands native bf16 here (no full-vocab fp32 buffer); chunks are upcast to fp32 downstream
        assert logits.dtype in (torch.float32, torch.bfloat16), f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    if logits.size(-1) > 1 and args.rollout_temperature > 0 and args.rollout_temperature != 1.0:
        logits = logits.div(args.rollout_temperature)
    if args.true_on_policy_mode:
        if getattr(args, "bf16", False):
            logits = logits.to(torch.bfloat16)
        elif getattr(args, "fp16", False):
            logits = logits.to(torch.float16)

    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
                logits_chunk = logits[start - 1 : end - 1]
            else:
                end += total_length
                start = end - response_length
                logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = parallel_state.cp.rank
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def get_responses_for_fused(
    hidden: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(hidden_chunk, tokens_chunk)` pairs for the fused path.

    Mirror of :func:`get_responses` for the fused-logprob path, where the input is
    decoder hidden states ``[1, T, H]`` rather than materialized logits
    ``[1, T, V]``. The CP/seq slicing is token-position-based (dim-0 only), so
    it applies identically to a ``[T, H]`` hidden tensor.

    Differences from :func:`get_responses`:

    * No temperature scaling — the fused kernel applies ``rollout_temperature``
      internally.
    * No true-on-policy dtype downcast — the fused path is incompatible with
      true-on-policy mode (asserted upstream).

    Kept as a standalone copy (not a shared helper) so the non-fused
    :func:`get_responses` stays byte-for-byte identical to upstream and merges
    cleanly.

    Args:
        hidden: Decoder hidden states with shape ``[1, T, H]``.
        args: Configuration.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.
        max_seq_lens: Max seq lens per sample (BSHD).

    Yields:
        Tuple of `(hidden_chunk, tokens_chunk)` where `hidden_chunk` is shape
        ``[R, H]`` and `tokens_chunk` is shape ``[R]`` (1D int64), both aligned
        to response tokens for one sample.
    """
    qkv_format = args.qkv_format

    assert len(hidden.shape) == 3, f"{hidden.shape}"

    if qkv_format == "thd":
        # Under context-parallel (CP>1) / dynamic-batch the decoder hidden arrives
        # as [S, B, H] (dim0 = per-rank token count, B==1) even when qkv_format is
        # "thd"; the output_layer would reshape it to the packed [1, T, H] form
        # internally before the (here-skipped) lm_head matmul. Replicate that here
        # so the thd packed-slicing below (end += total_length) sees the [T, H]
        # form and the CP token-position math stays aligned with the non-fused
        # logits path (which receives post-lm_head [1, T, V]). The size-1 batch dim
        # makes this a metadata-only reshape (no copy); B>1 would leave the tensor
        # unchanged and trip the assert below, failing loud rather than mis-slicing.
        if hidden.size(0) != 1 and hidden.size(1) == 1:
            hidden = hidden.reshape(1, -1, hidden.size(-1))
        assert hidden.size(0) == 1, f"{hidden.shape}"
        hidden = hidden.squeeze(0)
    else:
        assert max_seq_lens is not None
        hidden = hidden.view(-1, hidden.size(-1))

    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size

    if getattr(args, "sequence_parallel", False):
        # Under --sequence-parallel the decoder hidden arrives SP-sharded along
        # dim-0 (seq split by tp_size). The non-fused path gathers it to full
        # length inside the (here-skipped) output_layer.forward; replicate that
        # here so the CP/seq slicing below sees the full [T, H] layout. autograd
        # Function: fwd all-gather over TP, bwd reduce-scatter (no-op under TP=1,
        # since SP is force-disabled when TP=1, and a pure forward under no_grad
        # on the log-prob path).
        hidden = tensor_parallel.gather_from_sequence_parallel_region(
            hidden,
            tensor_parallel_output_grad=True,
            group=parallel_state.tp.group,
        )

    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
                hidden_chunk = hidden[start - 1 : end - 1]
            else:
                end += total_length
                start = end - response_length
                hidden_chunk = hidden[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns the
            # tensor for global positions [chunk_start, chunk_end).
            local_len = hidden.size(0)
            cp_rank = parallel_state.cp.rank
            chunk_start = cp_rank * local_len
            chunk_end = chunk_start + local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            global_start = resp_token_start - 1
            global_end = resp_token_end - 1

            s = max(global_start, chunk_start)
            e = min(global_end, chunk_end)
            if e <= s:
                hidden_chunk = hidden[0:0]
                tokens_chunk = tokens[0:0]
            else:
                hidden_chunk = hidden[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert hidden_chunk.size(0) == tokens_chunk.size(0), f"{hidden_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            chunk_0, chunk_1 = (
                hidden[end : end + chunk_size],
                hidden[end + chunk_size : end + 2 * chunk_size],
            )
            end += 2 * chunk_size

            chunk_0 = chunk_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            chunk_1 = chunk_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert chunk_0.size(0) == tokens_0.size(0), f"{chunk_0.size(0)} vs {tokens_0.size(0)}"
            assert chunk_1.size(0) == tokens_1.size(0), f"{chunk_1.size(0)} vs {tokens_1.size(0)}"

            hidden_chunk = torch.cat([chunk_0, chunk_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield hidden_chunk, tokens_chunk


def get_log_probs_and_entropy_for_fused(
    hidden: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    entropy_requires_grad: bool = True,
    max_seq_lens: list[int] | None = None,
    output_bias: torch.Tensor | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Fused-path log-probabilities (and optionally entropy) on responses.

    This is the fused (boundary C/A, ``args.fused_logprob``) counterpart of
    :func:`get_log_probs_and_entropy`: `hidden` is the decoder output
    ``[1, T, H]`` and `output_weight` is the TP-local lm_head weight shard
    ``[V_local, H]``. :func:`fused_linear_log_probs_and_entropy` fuses the
    lm_head matmul + log-prob so the ``[T, V]`` logits never materialize;
    only the ``[R]`` log-prob (and optional ``[R]`` entropy) is written back.

    Kept as a sibling entry point (not folded into
    :func:`get_log_probs_and_entropy`'s body) so the fused path is testable in
    isolation; :func:`get_log_probs_and_entropy` delegates here when
    ``args.fused_logprob`` is set and ``hidden``/``output_weight`` are provided.

    Args:
        hidden: Fused-path decoder hidden states ``[1, T, H]``.
        output_weight: Fused-path lm_head weight shard ``[V_local, H]``.
        args: Configuration (temperature applied inside `fused_logprob`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        entropy_requires_grad: If False, compute entropy as an observed metric
            without attaching it to the autograd graph.
        max_seq_lens: Max seq lens per sample (BSHD).
        output_bias: TP-local lm_head bias shard ``[V_local]`` or None. When
            provided the fused kernel computes ``log_softmax(hidden @ weight.T
            + bias)`` so the fused path matches the non-fused output_layer;
            None for the standard bias-less GPT head.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    parallel_state = get_parallel_state()
    log_probs_list: list[torch.Tensor] = []
    entropy_list: list[torch.Tensor] = []
    for chunk, tokens_chunk in get_responses_for_fused(
        hidden,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        # Per-sample fused lm_head + log-prob. The fused tile math (and, for the
        # triton/tilelang/cute_dsl backends, the kernel itself) dominates this
        # step; time it so the elapsed cost shows up in the perf metrics under
        # ``perf/fused_logprob_time`` (alongside ``perf/log_probs_time`` which
        # covers the whole forward_only pass). ``Timer.add`` accumulates into the
        # singleton Timer that ``log_perf_data_raw`` drains each step into
        # ``perf/<name>_time`` keys. The Timer resets per-step, so this reflects
        # the per-step fused-logprob cost (summed across samples / microbatches).
        # We bypass ``timer()`` / Timer.start/end to avoid the per-sample
        # "Timer fused_logprob start/end" log lines they emit on rank 0.
        #
        # Tile sizes: --fused-logprob-seq-chunk-size / --fused-logprob-vocab-chunk-size
        # (independent); -1 falls through to ``fused_logprob``'s built-in
        # defaults (seq=2048, vocab=4096).
        from time import perf_counter as _now

        _t0 = _now()
        log_prob, entropy = fused_linear_log_probs_and_entropy(
            chunk,
            output_weight,
            tokens_chunk,
            parallel_state.tp.group,
            bias=output_bias,
            temperature=args.rollout_temperature,
            with_entropy=with_entropy,
            entropy_requires_grad=entropy_requires_grad,
            seq_chunk_size=args.fused_logprob_seq_chunk_size,
            vocab_chunk_size=args.fused_logprob_vocab_chunk_size,
        )
        Timer().add("fused_logprob", _now() - _t0)

        # reshape instead of squeeze: when CP zigzag assigns exactly 1 response
        # token to this rank, squeeze(-1) on a [1] tensor produces a 0-d scalar
        # that torch.cat rejects downstream in log_rollout_data.
        log_probs_list.append(log_prob.reshape(-1))
        if with_entropy:
            entropy_list.append(entropy)

    res: dict[str, list[torch.Tensor]] = {"log_probs": log_probs_list}
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        allgather_cp_redistribute(
            res,
            logits=hidden,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    entropy_requires_grad: bool = True,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    hidden: torch.Tensor | None = None,
    output_weight: torch.Tensor | None = None,
    output_bias: torch.Tensor | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy is computed and returned only
    when requested.

    Two paths share this entry point:

    * **Non-fused** (default): `logits` is the materialized ``[1, T, V]`` output
      of the lm_head. `calculate_log_probs_and_entropy` runs the (Megatron)
      vocab-parallel cross-entropy. The ``[T, V]`` logits live in HBM.
    * **Fused** (boundary C/A, ``args.fused_logprob``): `hidden` is the decoder
      output ``[1, T, H]`` and `output_weight` is the TP-local lm_head weight
      shard ``[V_local, H]``. Delegates to
      :func:`get_log_probs_and_entropy_for_fused`, which fuses the lm_head
      matmul + log-prob so the ``[T, V]`` logits never materialize; only the
      ``[R]`` log-prob (and optional ``[R]`` entropy) is written back.

    Args:
        logits: Policy logits ``[1, T, V]`` on the non-fused path; a 0-d
            placeholder on the fused path (hidden/output_weight hold the data).
        args: Configuration (temperature applied inside `fused_logprob` on the
            fused path, inside `get_responses` otherwise).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        entropy_requires_grad: If False, compute entropy as an observed metric
            without attaching it to the autograd graph.
        non_loss_data: Unused; kept for API compatibility.
        max_seq_lens: Max seq lens per sample (BSHD).
        hidden: Fused-path decoder hidden states ``[1, T, H]``. None otherwise.
        output_weight: Fused-path lm_head weight shard ``[V_local, H]``. None
            otherwise.
        output_bias: Fused-path lm_head bias shard ``[V_local]``. None otherwise
            (the standard bias-less GPT head). Ignored on the non-fused path
            (the materialized logits already include the bias).

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    assert non_loss_data
    if bool(getattr(args, "fused_logprob", False)) and hidden is not None and output_weight is not None:
        return get_log_probs_and_entropy_for_fused(
            hidden,
            output_weight,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=with_entropy,
            entropy_requires_grad=entropy_requires_grad,
            max_seq_lens=max_seq_lens,
            output_bias=output_bias,
        )

    parallel_state = get_parallel_state()
    log_probs_list = []
    entropy_list = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            parallel_state.tp.group,
            with_entropy=with_entropy,
            entropy_requires_grad=entropy_requires_grad,
            chunk_size=args.log_probs_chunk_size,
            true_on_policy=args.true_on_policy_mode,
            vocab_size=getattr(args, "vocab_size", None),
        )

        # reshape instead of squeeze: when CP zigzag assigns exactly 1 response
        # token to this rank, squeeze(-1) on a [1] tensor produces a 0-d scalar
        # that torch.cat rejects downstream in log_rollout_data.
        log_probs_list.append(log_prob.reshape(-1))
        if with_entropy:
            entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    entropy_requires_grad: bool = False,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.
        entropy_requires_grad: Unused; kept for signature compatibility.
            ``forward_only`` binds it unconditionally onto both the log-prob
            and the value callback; the value path ignores it (values carry
            no entropy).

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        # upcast (no-op for fp32) so value-head outputs stay fp32 even when logits arrive bf16
        value_list.append(logits_chunk.squeeze(-1).float())

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res
