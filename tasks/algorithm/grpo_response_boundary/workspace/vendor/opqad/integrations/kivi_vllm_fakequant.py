"""KIVI (arXiv:2402.02750) as an accuracy-faithful baseline inside vLLM.

Why fake quantization and not a packed cache
--------------------------------------------
The baseline table needs KIVI's *accuracy* on our models, our benchmark and our
shape.  It does not need KIVI's memory footprint -- we are not claiming KIVI's
throughput, we are claiming OSCAR's, and that is measured separately.

A real KIVI cache means a new paged layout (per-channel key scales are per
*block*, not per token, so they do not fit the per-token slot the OSCAR path
uses) plus a decode kernel that reads it.  That is a kernel project.  Quantizing
and immediately dequantizing in the BF16 cache produces bit-identical *values*
to what a packed KIVI cache would hand the attention kernel, at zero kernel
cost.  The arm is therefore exact on the axis it is being read for and silent on
the axis it is not -- which is the honest trade, as long as the table says so.

What KIVI actually is
---------------------
Two asymmetric min/max quantizers with different axes, which is the whole point
of the paper:

* **Keys: per channel, over a group of G tokens.**  Key outliers live in fixed
  channels, so sharing a scale down the channel is what destroys them and
  sharing it across tokens within a channel is what does not.
* **Values: per token, over head_dim.**  Value outliers are not channel-aligned.
* **A full-precision residual** of the most recent tokens, because the newest
  group is not full yet and quantizing a partial group is both wrong and
  unnecessary.

Mapping that onto a paged cache
-------------------------------
``SimpleKVConfig`` in the vLLM tree already states the constraint: a key group
must equal the paged block size, or a group straddles two blocks, which are not
contiguous in memory.  So we set ``G = block_size`` and quantize a block exactly
once, at the moment it fills.

That is not an approximation of KIVI -- it *is* KIVI's own schedule.  KIVI
buffers the newest tokens in fp16 and flushes a group when it is complete; here
the buffer is the partial block, so the residual length runs 0..G-1 with mean
G/2.  Run with ``VLLM_OSCAR_BLOCK_SIZE=32`` to get the paper's G=32.

Idempotence is what makes this safe.  Asymmetric min/max quantization maps the
block's min and max onto grid points and every other value onto a grid point, so
re-quantizing an already-quantized block is a no-op up to fp rounding.  Missing
a block, or quantizing one twice, cannot corrupt the cache; the per-step
bookkeeping below is an optimization, not a correctness requirement.

Eager only
----------
The hook runs Python per layer per step.  Under a captured CUDA graph the
capture would bake in one step's block indices and replay them forever, which is
exactly the class of bug that made every OSCAR arm read ~0 for months.  So this
refuses to install unless ``enforce_eager`` is set, rather than silently
producing a number that is not KIVI.
"""
from __future__ import annotations

import os

from opqad.utils.envreg import env_bool, env_int

_applied = False


def enabled() -> bool:
    return env_bool("QAT_KIVI_FAKEQUANT")


def _qdq(x, dim: int, levels: int):
    """Asymmetric min/max quantize-dequantize along ``dim``.

    A constant group has no representable range.  ``scale -> 1`` there keeps the
    exact value, which is what KIVI's CUDA kernel does with the same metadata;
    clamping the scale to an epsilon instead would divide by that epsilon and
    manufacture noise out of a group that had none.  Same treatment as
    ``opqad/quantization/kivi_fake_quant.py``, deliberately -- the two implementations are
    cross-checked against each other in tests/test_kivi_vllm_fakequant.py.
    """
    import torch

    lo = x.amin(dim=dim, keepdim=True)
    hi = x.amax(dim=dim, keepdim=True)
    scale = (hi - lo) / levels
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    return ((x - lo) / safe).clamp(0, levels).round() * safe + lo


def kv_views(kv_cache, num_kv_heads: int, head_dim: int):
    """Return ``(key_view, value_view, token_axis, block_size)``.

    vLLM's backends disagree about the cache's shape, and the disagreement is
    not cosmetic -- it moves the token axis, which is the axis KIVI's key rule
    reduces over.  Reducing over the wrong one produces a valid-looking 2-bit
    grid and a completely different method.

        FlashAttention   [2, nb, block, Hk, D]        key = cache[0]
        FlashInfer NHD   [nb, 2, block, Hk, D]        key = cache[:, 0]
        FlashInfer HND   [nb, 2, Hk, block, D]        key = cache[:, 0]

    Both views index blocks along dim 0 and are writable through to the base
    storage, so ``index_copy_`` on them updates the real cache.
    """
    s = tuple(kv_cache.shape)
    if len(s) != 5 or s[-1] != head_dim:
        raise ValueError(f"unrecognised KV cache shape {s} for D={head_dim}")
    # Collect every reading the shape admits, then insist there is exactly one.
    # Hk == block_size makes NHD and HND indistinguishable from the shape alone,
    # and picking one would put the key rule's reduction on the wrong axis --
    # which yields a valid 2-bit grid and a different method.  None of the
    # models here are ambiguous (Hk is 2 or 8 against block 32); if one ever is,
    # this needs the backend's declared layout, not a guess.
    cands = []
    if s[0] == 2 and s[3] == num_kv_heads:
        cands.append(("FA", kv_cache[0], kv_cache[1], 1, s[2]))
    if s[1] == 2 and s[3] == num_kv_heads:
        cands.append(("NHD", kv_cache[:, 0], kv_cache[:, 1], 1, s[2]))
    if s[1] == 2 and s[2] == num_kv_heads:
        cands.append(("HND", kv_cache[:, 0], kv_cache[:, 1], 2, s[3]))
    if len(cands) != 1:
        raise ValueError(
            f"KV cache shape {s} with Hk={num_kv_heads}, D={head_dim} matches "
            f"{[c[0] for c in cands] or 'no'} layout(s); refusing to guess "
            "which axis holds the tokens"
        )
    _, k, v, tax, block = cands[0]
    return k, v, tax, block


def _quantize_blocks(kv_cache, ids, bits: int, value_group: int,
                     num_kv_heads: int | None = None,
                     head_dim: int | None = None) -> None:
    """Quantize-dequantize the given blocks in place, KIVI's two rules.

    ``ids`` indexes blocks.  ``num_kv_heads``/``head_dim`` select the layout;
    omitted, the FlashAttention layout ``[2, nb, block, Hk, D]`` is assumed,
    which is what the unit tests build.
    """
    if ids.numel() == 0:
        return
    levels = (1 << bits) - 1
    if num_kv_heads is None or head_dim is None:
        kc, vc, tax = kv_cache[0], kv_cache[1], 1
    else:
        kc, vc, tax, _ = kv_views(kv_cache, num_kv_heads, head_dim)

    # Keys: per channel, shared across the block's tokens.  Reducing over the
    # token axis is the whole of "per channel over a group of G tokens", and it
    # is the reason KIVI survives 2 bits where per-token quantization does not:
    # key outliers sit in fixed channels, so a scale that runs down a channel
    # never has to span one.
    k = kc.index_select(0, ids).float()
    kc.index_copy_(0, ids, _qdq(k, tax, levels).to(kv_cache.dtype))

    # Values: per token, in channel groups of ``value_group``.  Value outliers
    # are not channel-aligned, so the axis flips.  Grouping along head_dim
    # rather than taking it whole follows KIVI's released kernel.  head_dim is
    # last in every layout above, so this one needs no layout branch.
    v = vc.index_select(0, ids).float()
    d = v.shape[-1]
    if d % value_group:
        raise ValueError(f"head_dim {d} not divisible by value group {value_group}")
    vq = _qdq(v.reshape(*v.shape[:-1], d // value_group, value_group), -1, levels)
    vc.index_copy_(0, ids, vq.reshape(v.shape).to(kv_cache.dtype))


# Every v1 backend vLLM might pick for a BF16 cache on this hardware.  Patching
# only one is how the first version of this module produced a "KIVI" result that
# was plain BF16: the selector logged
#   "Using FLASHINFER attention backend out of potential backends:
#    ['FLASHINFER', 'FLASH_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION']"
# and the hook was on FlashAttentionImpl, which was never called.  The run
# finished, wrote `kv_cache_quantization: kivi_2bit_fakequant`, and measured
# nothing.  Patch them all and assert at runtime that one of them fired.
_BACKENDS = (
    ("vllm.v1.attention.backends.flash_attn", "FlashAttentionImpl"),
    ("vllm.v1.attention.backends.flashinfer", "FlashInferImpl"),
    ("vllm.v1.attention.backends.triton_attn", "TritonAttentionImpl"),
)


def maybe_apply(verbose: bool = True) -> bool:
    """Wrap every v1 attention backend so filled blocks are KIVI-quantized."""
    global _applied
    if _applied or not enabled():
        return False

    try:
        import importlib

        import torch
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[kivi] not applied: {type(exc).__name__}: {exc}", flush=True)
        return False

    bits = env_int("QAT_KIVI_BITS")
    # KIVI's released kernel groups values along head_dim in 32s.
    value_group = env_int("QAT_KIVI_VALUE_GROUP")
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"QAT_KIVI_BITS must be 2/3/4/8, got {bits}")

    # One entry per attention-metadata object: the block ids that completed on
    # this step.  Every layer in a step sees the same metadata object, so the
    # (synchronizing) index computation happens once per step, not once per
    # layer per step.
    step_cache: dict[int, object] = {}
    state = {"calls": 0, "blocks": 0, "logged": False}

    def _completed_blocks(attn_metadata, block_size):
        key = id(attn_metadata)
        hit = step_cache.get(key)
        if hit is not None:
            return hit
        step_cache.clear()

        # `slot_mapping` is the one field every v1 backend carries with the same
        # meaning: the flat cache slot each token of this forward was written
        # to, where slot = block_id * block_size + offset.  Deriving the filled
        # blocks from it rather than from seq_lens/block_table makes this
        # independent of the backends' disagreements about metadata -- and
        # handles prefill (many blocks at once) and decode (at most one) with
        # the same two lines.
        slots = getattr(attn_metadata, "slot_mapping", None)
        if slots is None:
            return None
        slots = slots.reshape(-1)
        # Padded positions carry -1.  A block is full exactly when its last
        # offset was written; a sequence that stops earlier leaves its final
        # block partial, which is KIVI's full-precision residual.
        done = slots[(slots >= 0) & (slots % block_size == block_size - 1)]
        ids = torch.div(done, block_size, rounding_mode="floor").to(torch.long)
        ids = ids.unique()
        step_cache[key] = ids
        return ids

    def _wrap(original, backend: str):
        def _forward(self, layer, query, key, value, kv_cache, attn_metadata,
                     output, *a, **kw):
            out = original(self, layer, query, key, value, kv_cache,
                           attn_metadata, output, *a, **kw)
            # No cache (profiling run, encoder-only attention): nothing to do.
            if attn_metadata is None or kv_cache is None or kv_cache.numel() == 0:
                return out
            if torch.cuda.is_current_stream_capturing():
                # A capture would freeze this step's block indices into the
                # graph and replay them forever, so the arm would quantize the
                # wrong blocks while printing a correct-looking banner.  That is
                # how every OSCAR arm read ~0 for months; refuse instead.
                raise RuntimeError(
                    "[kivi] the fake quantizer is being CUDA-graph captured. "
                    "Run this arm eager (EVAL_ENFORCE_EAGER=1): the hook picks "
                    "blocks per step, and a captured graph would replay one "
                    "step's picks for the whole run."
                )
            state["calls"] += 1
            try:
                hk, d = key.shape[-2], key.shape[-1]
                _, _, tax, block_size = kv_views(kv_cache, hk, d)
                ids = _completed_blocks(attn_metadata, block_size)
                if ids is None:
                    return out
                _quantize_blocks(kv_cache, ids, bits, value_group, hk, d)
                state["blocks"] += int(ids.numel())
                if not state["logged"] and state["blocks"]:
                    state["logged"] = True
                    print(f"[kivi] fake quant live on {backend}: bits={bits}, "
                          f"key group = block_size = {block_size} tokens "
                          f"(KIVI's G is 32; set VLLM_OSCAR_BLOCK_SIZE=32), "
                          f"token axis {tax}, values per token in channel "
                          f"groups of {value_group}, residual = the unfilled "
                          f"block", flush=True)
            except Exception as exc:  # noqa: BLE001
                # Failing open would report a BF16 run as a KIVI run.
                raise RuntimeError(
                    f"[kivi] fake quant failed on a live batch "
                    f"({backend}, kv_cache {tuple(kv_cache.shape)}, "
                    f"key {tuple(key.shape)}): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            # The failure that produced a BF16 run wearing KIVI's label was
            # silent because nothing checked that the hook ever did anything.
            # Decode reaches a block boundary every `block_size` steps, so by
            # a few thousand attention calls it is not possible to have
            # quantized nothing unless the wrong thing is being wrapped.
            if state["blocks"] == 0 and state["calls"] > 4096:
                raise RuntimeError(
                    f"[kivi] {state['calls']} attention calls on {backend} and "
                    "not one block quantized. The arm is running unquantized "
                    "BF16; refusing to report it as KIVI."
                )
            return out

        return _forward

    patched = []
    for mod_name, cls_name in _BACKENDS:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
        except Exception:  # noqa: BLE001, S112
            continue    # backend not built in this install; not an error
        cls.forward = _wrap(cls.forward, cls_name)
        patched.append(cls_name)

    if not patched:
        raise RuntimeError(
            "[kivi] no v1 attention backend could be wrapped; the arm would "
            "run as plain BF16. Checked: "
            + ", ".join(c for _, c in _BACKENDS)
        )

    _applied = True
    if verbose:
        print(f"[kivi] KIVI-{bits}bit fake quantization ON, wrapping "
              f"{', '.join(patched)}. Keys per channel over one paged block, "
              f"values per token, the unfilled block left in full precision. "
              f"Accuracy-faithful; memory and throughput are NOT KIVI's -- the "
              f"cache stays BF16.", flush=True)
    return True
