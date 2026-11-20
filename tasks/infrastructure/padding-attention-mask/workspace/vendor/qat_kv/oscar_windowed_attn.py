"""Serving-faithful OSCAR attention for the training forward.

The problem
-----------
At serving, OSCAR keeps the first ``sink`` tokens and the last ``recent`` tokens
*relative to the current decode position* in BF16, and reads everything else
from the INT2 cache.  The training fake quantizer could not reproduce that --
under teacher forcing every query position has a different recent window -- so
it quantized everything and documented the omission as "the safe direction: the
student trains against a harsher quantizer than it is served with".

Harsher, yes.  Safe, no.  Measured on real post-RoPE activations at 4096 context
(experiments/2026-09-15/scripts/measure_recent_window_divergence.py):

    model   nmse(train, recent ignored)   nmse(serve, recent BF16)   ratio
    1.5B              0.12886                     0.01654            7.8x
    4B                0.13601                     0.02543            5.3x

So 87% (1.5B) / 81% (4B) of the error the training loss is built on comes from
tokens that deployment never quantizes.  That is not a harder version of the
deployment problem, it is a different one -- and it lands exactly where this
model keeps its attention: the last 256 tokens carry 71-95% of the 1.5B's mass
(measure_attention_mass.py).

The fix
-------
Two attentions over the same Q, with complementary masks, merged by
log-sum-exp -- which is what the serving path already does at decode time
(``_decode_attention_windowed`` merges an INT2 branch and a staged-BF16 branch
through exactly this ``logaddexp``).

    exact branch      raw K/V,       mask = causal AND (q - k < recent OR k < sink)
    quantized branch  fake-quant K/V, mask = causal AND NOT the above

Done naively this needs the [T, T] logits the old docstring called infeasible,
and at 16k that is 34 GB of fp32 per layer.  ``flex_attention`` avoids it: the
masks are block-sparse, so the exact branch only visits the ~``recent``-wide
band plus the sink columns, and ``return_lse=True`` hands back the log-sum-exp
that the merge needs without ever materializing the scores.  The quantized
branch skips the band it does not own.

Cost, measured rather than assumed (``bench_windowed_attn.py``, job 57703):

    shape           attention only        + quant/dequant     total
    1.5B T=4096     0.046 -> 0.368 ms     0.527 ms (both)     1.56x
    1.5B T=16384    0.443 -> 2.036 ms     0.848 ms            2.23x
    4B   T=16384    1.175 -> 4.968 ms     2.696 ms            1.98x

An earlier draft of this note claimed "close to one causal attention, not two".
That was wrong: the attention itself is 4-8x, and only the quantize/dequantize
both paths already pay brings the combined figure down to ~2x.  Attention is not
the whole training step, so the end-to-end cost is lower again -- but that has
not been measured, and should be before a long arm commits to this.

Numerics
--------
The merge is exact, not an approximation: for disjoint key sets A and B,

    softmax over A∪B  =  w_A * softmax_A + w_B * softmax_B,
    w_A = exp(lse_A - lse_AB),  lse_AB = logaddexp(lse_A, lse_B)

Rows where a branch owns no keys get ``lse = -inf`` and drop out with weight 0.
``verify_against_reference`` in the test checks this against a dense O(T^2)
reimplementation.

Gated on ``QAT_OSCAR_WINDOWED_ATTN=1``.  Default off: it changes what the loss
optimizes, so it is a different experiment, not a bug fix to an existing arm.
"""
from __future__ import annotations

import functools
import collections
import weakref
import os
from typing import Optional

import torch

__all__ = ["enabled", "require_for_oscar", "windowed_fake_quant_attention"]


def enabled() -> bool:
    return os.environ.get("QAT_OSCAR_WINDOWED_ATTN", "").strip() == "1"


def require_for_oscar() -> None:
    """OSCAR attends through the window. Always. This is not a knob.

    The sink/recent window is not an option OSCAR has, it is what OSCAR *is*:
    the serving path keeps sink 64 + recent 256 per request in BF16, and a
    training loss built without it is built on a reconstruction error 5.3-7.8x
    larger than deployment's.  For a while this repo exposed the choice as
    ``QAT_OSCAR_WINDOWED_ATTN`` defaulting to 0, and it cost:

      * ``qat_e2e.sh:192`` reading ``${QAT_OSCAR_WINDOWED_ATTN:-0}``, so every
        arm launched through it silently dropped the window from the loss and
        nothing in the banner said so;
      * the non-windowed fall-through in ``turboquant_attn_hook`` quantizing
        through ``_quantize``, which honours ``sink_tokens`` as an ABSOLUTE
        column index -- and verl left-pads prompts to 512, so that sink
        protected padding rather than the first real tokens.

    Both are the same failure the skip-layer knob was
    (``turboquant_fake_quant.oscar_quant_all_layers``): a default quietly
    deciding which experiment ran.  So the choice is gone for OSCAR.  The
    variable is still *read*, and only to refuse -- an arm that leaves it unset
    or passes 0 stops loudly instead of training a different objective.

    k2v3, the INT4 schemes and the TurboQuant tiers are untouched: they have no
    sink/recent window to reproduce, so this says nothing about them.
    """
    if enabled():
        return
    raw = os.environ.get("QAT_OSCAR_WINDOWED_ATTN")
    raise RuntimeError(
        "QAT_OSCAR_WINDOWED_ATTN="
        + ("<unset>" if raw is None else repr(raw))
        + " but the quantizer is OSCAR, which attends through its sink/recent "
        "window by definition. Set QAT_OSCAR_WINDOWED_ATTN=1.\n"
        "Without it the loss is built on an error 5.3-7.8x larger than "
        "deployment's, and the fall-through path applies sink_tokens as an "
        "ABSOLUTE column index -- which, against verl's left-padded prompts, "
        "protects padding instead of real tokens.\n"
        "This is not a performance switch and there is no non-windowed OSCAR."
    )


@functools.lru_cache(maxsize=1)
def _flex():
    """``flex_attention``, compiled.

    Uncompiled, flex_attention runs an eager fallback and warns that it will be
    slow -- and slow here is not a detail, because this replaces the attention
    of every quantized layer of every training step.  Compiling is the
    documented path and is what makes the block-sparse masks actually skip
    work: the exact branch should touch only the ~recent-wide band plus the
    sink columns, which an eager implementation cannot exploit.

    Compiled once and cached.  If compilation fails (an old Inductor, a cold
    cache on a node with no writable TRITON_CACHE_DIR) fall back to the eager
    callable rather than failing the run: the numerics are identical, only the
    speed differs, and a correctness-preserving fallback is the right trade for
    a training job that has already started.
    """
    from torch.nn.attention.flex_attention import flex_attention

    # Dynamo gives up after `recompile_limit` cache entries for a frame and then
    # runs it eagerly FOREVER -- silently, apart from one warning line.
    # flex_attention's eager path materializes the full [T, T] score matrix, so
    # that fallback is not a small regression: it measured 234x a plain causal
    # attention at T=4096, and it is the thing that OOM'd the logitdist dumps.
    #
    # MEASURED cost in cache entries: exactly TWO per distinct sequence length,
    # one for the bm_exact call and one for the bm_quant call (probe:
    # scripts/logitdist/probe_recompile.py -- +2 frames for every T).
    #
    # The old 64 was sized for a TRAINING step, which legitimately sees one
    # shape plus the odd micro-batch tail.  A teacher-forced dump is not that
    # workload: it walks one trajectory per distinct T, so 3 trajectory labels x
    # 20 samples = 60 lengths = 120 entries, and 64 was reached at ~32 lengths.
    # That is exactly what the 4B and 8B dumps hit -- they fell into the eager
    # path mid-run and died on a 127.96 GiB [1,32,32768,32768] fp32 score
    # matrix.  The 1.5B hit the same limit but survived it, because at H=12 the
    # eager matrix is ~51 GiB and fits in 275 GiB of HBM.
    #
    # Raising the limit cannot change numerics: the compiled and eager paths
    # compute the same attention, they differ in memory and speed.  What it
    # changes is that the intended (compiled, tiled) path keeps being used.
    #
    # `accumulated_recompile_limit` (default 256) is a SECOND wall on the same
    # frame -- `exceeds_recompile_limit` also returns True once
    # `compile_id.frame_compile_id` reaches it -- so raising only the first
    # would move the fallback rather than remove it.
    try:
        import torch._dynamo.config as _dynamo_cfg

        if getattr(_dynamo_cfg, "recompile_limit", 8) < 256:
            _dynamo_cfg.recompile_limit = 256
        if getattr(_dynamo_cfg, "accumulated_recompile_limit", 256) < 1024:
            _dynamo_cfg.accumulated_recompile_limit = 1024
    except Exception:  # noqa: BLE001
        pass

    try:
        return torch.compile(flex_attention, dynamic=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[oscar-windowed] torch.compile(flex_attention) failed, using "
              f"the eager fallback: {type(exc).__name__}: {exc}", flush=True)
        return flex_attention


def _padded_len(n: int, T: int) -> int:
    """The length a compacted row is padded to before the recursive call.

    ONE length for the whole run: the full sequence length ``T`` this call was
    invoked with.  Every row of every micro-batch at a given ``T`` therefore
    compiles ``flex_attention`` once.

    The alternative -- rounding each row up to the next power-of-two bucket --
    is what produced the crash this replaces.  Rows have different lengths, so
    bucketing gave a different padded length per row (measured: 8 rows -> 7
    distinct lengths), each one a separate dynamo specialization of the same
    frame.  Thirty-six layers reached ``torch._dynamo``'s ``recompile_limit``
    (64), dynamo abandoned the frame and ran it eagerly from then on, and a
    forward and its checkpointed recomputation that straddled the switch saved
    different tensors -- ``CheckpointError`` at step 2-3 of j114812 and j114918,
    both 4B + OSCAR.  The 1.5B (28 layers) and k2v3 arms never accumulated
    enough specializations to cross the limit.

    Measured cost of padding to ``T`` instead: a row of 13312 tokens in a
    16896 batch carries ~140 MiB more per layer (5 tensors at H=32 D=128 bf16),
    ~5 GiB across 36 layers -- against an activation bug that kills the run.
    Coarser fixed buckets would cost less but reintroduce more than one shape;
    the whole point is that there is exactly one.

    ``n > T`` cannot happen -- ``n`` counts valid tokens within a row of length
    ``T`` -- but the max() keeps this total if a caller ever passes something
    else.
    """
    return max(T, n)


@functools.lru_cache(maxsize=32)
def _block_masks(T: int, sink: int, recent: int, device_str: str):
    """Block masks for the exact branch and its complement.

    Cached on (T, sink, recent): every layer of every step at a given sequence
    length reuses them, and building a BlockMask is not free.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    def exact_mask(b, h, q, k):
        return (k <= q) & (((q - k) < recent) | (k < sink))

    def quant_mask(b, h, q, k):
        return (k <= q) & ~(((q - k) < recent) | (k < sink))

    dev = torch.device(device_str)
    # B=None, H=None -> shared across batch and heads, which is what these are.
    return (
        create_block_mask(exact_mask, None, None, T, T, device=dev),
        create_block_mask(quant_mask, None, None, T, T, device=dev),
    )


# Cache by live tensor identity and mutation version. Weak references avoid
# retaining HF's existing quadratic additive mask. The cache itself owns only
# O(B*T) compact-token indices, never an extra [T,T] mask.
_padding_masks = collections.OrderedDict()


def _mask_version(mask):
    try:
        return mask._version
    except RuntimeError:  # inference-mode tensors have no version counter
        return None


def _valid_keys(attention_mask, B, T, device):
    """Extract padding from the supported HF causal-mask contract.

    Supports [B,T] binary validity and HF [B,1,T,T] / [B,1,1,T]
    boolean-allowed or additive masks (0 allowed, -inf/dtype.min blocked).
    A four-dimensional input MUST be standard causal + padding, without
    additional sliding windows, document masks, or attention biases. Its last
    query row exposes all valid keys. The caller uses square teacher forcing.
    We inspect only O(B*T) entries; validation of arbitrary quadratic masks is
    deliberately outside this interface. No extra finite attention bias is
    applied by this function.
    """
    if attention_mask.device != device:
        raise ValueError("OSCAR attention_mask must be on the query device")
    if attention_mask.ndim == 2:
        if tuple(attention_mask.shape) != (B, T):
            raise ValueError(f"expected binary attention_mask {(B,T)}, got {tuple(attention_mask.shape)}")
        if attention_mask.dtype == torch.bool:
            return attention_mask
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("2D attention_mask must be binary validity, not additive bias")
        return attention_mask != 0
    if attention_mask.ndim != 4 or attention_mask.shape[:2] != (B, 1) or attention_mask.shape[-1] != T or attention_mask.shape[-2] not in (1, T):
        raise ValueError("OSCAR expects 2D binary or standard HF 4D causal/padding mask")
    row = attention_mask[:, 0, -1, :]
    if row.dtype == torch.bool:
        return row
    if not row.dtype.is_floating_point:
        raise ValueError("4D attention_mask must be bool-allowed or floating additive")
    blocked = torch.isneginf(row) | (row == torch.finfo(row.dtype).min)
    if not bool(((row == 0) | blocked).all()):
        raise ValueError("finite attention bias is unsupported; use standard HF causal/padding mask")
    return row == 0


def _padding_layout(attention_mask, B, T, device):
    """Cache O(B*T) compact-token indices, without keeping a 4D HF mask alive."""
    if attention_mask is None:
        return None
    cache_key = (id(attention_mask), _mask_version(attention_mask), B, T, str(device))
    cached = _padding_masks.get(cache_key)
    if cached is not None and cached[0]() is attention_mask:
        _padding_masks.move_to_end(cache_key)
        return cached[1]
    valid = _valid_keys(attention_mask, B, T, device)
    result = None if bool(valid.all()) else tuple(
        torch.nonzero(valid[b], as_tuple=False).flatten() for b in range(B)
    )
    _padding_masks[cache_key] = (weakref.ref(attention_mask), result)
    _padding_masks.move_to_end(cache_key)
    while len(_padding_masks) > 4:
        _padding_masks.popitem(last=False)
    return result


def windowed_fake_quant_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_q: torch.Tensor,
    value_q: torch.Tensor,
    *,
    sink: int,
    recent: int,
    scale: float,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention where the sink+recent window reads raw K/V and the rest reads quantized.

    All tensors are ``[B, H, T, D]`` with ``H`` already repeated to the query
    head count (flex_attention's ``enable_gqa`` is not used here: the caller's
    quantized tensors are built on un-repeated KV heads and repeating them once
    is cheaper than threading the group size through two mask closures).

    Returns ``[B, H, T, D]``, same dtype as ``query``.
    """
    flex_attention = _flex()

    T = query.shape[-2]
    if key.shape[-2] != T:
        raise NotImplementedError(
            "windowed fake-quant attention needs a square self-attention forward "
            f"(q_len == kv_len); got q_len={T} kv_len={key.shape[-2]}.  The "
            "training forward is teacher-forced and always square; a cached-KV "
            "forward would need the decode-time formulation instead."
        )

    layout = _padding_layout(attention_mask, query.shape[0], T, query.device)
    if layout is not None:
        # Reference-correct path: evaluate each real sequence on the original
        # no-padding kernel, then scatter its output back. This naturally makes
        # sink indices and recent distances relative to actual sequence tokens.
        # Padding positions stay zero and never enter any attention softmax.
        # The original compiled kernels are reused by real sequence length;
        # no B*T*T mask or changed attention approximation is introduced.
        outputs = []
        tensors = (query, key, value, key_q, value_q)
        for row, indices in enumerate(layout):
            if indices.numel() == 0:
                # Connect every input to a zero result so an entirely padded
                # batch still has finite zero gradients rather than None.
                zero = sum(t[row:row+1].sum() * 0 for t in tensors[1:])
                outputs.append(query[row:row+1] * 0 + zero)
                continue
            compact = tuple(t[row:row+1].index_select(-2, indices) for t in tensors)
            # Pad every row to ONE length for the whole run, not to a per-row
            # bucket.
            #
            # The old rule rounded each row up to the next 1024.  Rows have
            # different lengths, so that produced a different padded length per
            # row -- measured, 8 rows gave 7 distinct lengths (2048, 4096,
            # 10240, 13312, 16384, 17408, 33792).  Each distinct length is a
            # separate dynamo specialization of the SAME flex_attention frame,
            # and 36 layers of that exhausts recompile_limit (64), at which
            # point dynamo gives up and runs the frame eagerly FOREVER.  The
            # warning is one line and easy to miss; the consequence is not.
            #
            # If the switch to eager lands between a forward and its
            # checkpointed recomputation, the two executions save a different
            # set of tensors and backward dies with
            #
            #   torch.utils.checkpoint.CheckpointError:
            #   Recomputed values ... have different metadata
            #
            # which is exactly how j114812 and j114918 (4B + OSCAR) failed, at
            # step 2-3, while 1.5B + OSCAR and 4B + k2v3 -- fewer layers, fewer
            # specializations -- ran fine.
            #
            # A single fixed length means one shape, one compilation, no
            # fallback, and identical behaviour forward and recomputed.  The
            # padding is still only future (right) zeros, so causal real-token
            # outputs and gradients cannot read it; this is not left padding and
            # not a moved sink.
            n = indices.numel()
            # Pad to the length this row will be BROUGHT TO anyway -- see
            # _padded_len.  Note the mask is deliberately NOT forwarded: after
            # index_select the row has no padding left, and passing the original
            # mask would make the recursion re-enter this same branch (the
            # indices would no longer match the compacted length).
            target = _padded_len(n, T)
            if target != n:
                compact = tuple(torch.nn.functional.pad(t, (0, 0, 0, target-n)) for t in compact)
            compact_out = windowed_fake_quant_attention(
                *compact, sink=sink, recent=recent, scale=scale
            )[:, :, :n, :]
            outputs.append(torch.zeros_like(query[row:row+1]).index_copy(2, indices, compact_out))
        return torch.cat(outputs, dim=0)

    bm_exact, bm_quant = _block_masks(T, int(sink), int(recent), str(query.device))

    o1, lse1 = flex_attention(
        query, key, value, block_mask=bm_exact, scale=scale, return_lse=True
    )
    o2, lse2 = flex_attention(
        query, key_q, value_q, block_mask=bm_quant, scale=scale, return_lse=True
    )

    # A row whose branch owns no keys comes back with lse = -inf and garbage in
    # the output; nan_to_num after weighting by exp(-inf - finite) = 0 would
    # still propagate a nan through 0 * nan, so clean the outputs first.
    o1 = torch.nan_to_num(o1)
    o2 = torch.nan_to_num(o2)
    neg_inf = torch.finfo(lse1.dtype).min
    lse1 = torch.where(torch.isfinite(lse1), lse1, torch.full_like(lse1, neg_inf))
    lse2 = torch.where(torch.isfinite(lse2), lse2, torch.full_like(lse2, neg_inf))

    lse = torch.logaddexp(lse1, lse2)
    w1 = torch.exp(lse1 - lse).unsqueeze(-1)
    w2 = torch.exp(lse2 - lse).unsqueeze(-1)
    out = (o1 * w1 + o2 * w2).to(query.dtype)
    _warn_if_eager()
    return out


_warned = False


def _warn_if_eager() -> None:
    """Say so, once, if dynamo has given up and this is running eagerly.

    The failure mode this guards is the one that has cost the most time on this
    project: something silently degrades to a slower or less accurate path and
    nothing in the log changes.  Here the degradation is ~200x, so a run that
    hits it looks like a hang rather than a slowdown, and the operator has no
    way to tell it from a scheduling problem.
    """
    global _warned
    if _warned:
        return
    try:
        from torch._dynamo.utils import counters

        if counters["frames"].get("ok", 0) == 0 and counters["frames"].get("total", 0):
            _warned = True
            print(
                "[oscar-windowed] WARNING: flex_attention is NOT compiled -- dynamo "
                "compiled 0 of "
                f"{counters['frames'].get('total')} frames.  The eager path "
                "materializes the full [T, T] score matrix and measured ~200x a "
                "plain causal attention.  Look for 'recompile_limit' in stderr.",
                flush=True,
            )
    except Exception:  # noqa: BLE001
        pass
