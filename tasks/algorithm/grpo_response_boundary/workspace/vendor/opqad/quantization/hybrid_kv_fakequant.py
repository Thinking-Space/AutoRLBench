"""The hybrid 2-bit KV arm: per-channel K, OSCAR's V, OSCAR's BF16 window.

Why this arm exists
-------------------
The 32k AIME24 table has OSCAR INT2 and KIVI trading places by model --
91.6% vs 76.8% on Qwen2-1.5B, 83.7% vs 91.3% on Qwen3-4B -- and the offline
probe (`experiments/2026-09-17/scripts/kv_quant_nmse_probe.py`) says why the
two are not simply ordered:

    middle-region NMSE, 2 bits     K        V
      OSCAR (rotated, per token)   0.142    0.126
      KIVI  (per channel, G=32)    0.039    0.178

Each scheme owns one axis.  KIVI's per-channel key grid is 2.6-3.7x more
accurate than OSCAR's rotated per-token one, and the same probe shows rotating
*before* a per-channel grid is slightly worse than not rotating -- R_k equalises
a token's range across channels, which is what the per-token grid needs and what
the per-channel grid does not.  OSCAR's value rule is the better half by 1.4x.
And OSCAR keeps a BF16 sink and recent window that KIVI has no equivalent of,
which is the only reason OSCAR wins anywhere despite the worse K: NMSE does not
predict accuracy, and on the 1.5B OSCAR wins end to end with 2.6x the K error.

So this module is the union rather than a swap:

    K   per channel over one paged block          (KIVI's axis, no rotation)
    V   rotate by R_v, per token over head_dim,   (OSCAR's rule, verbatim --
        percentile clip, un-rotate                 it calls the same function
                                                   the training path calls)
    +   sink 64 and recent 256 tokens per request left in BF16 (OSCAR's window)

Fake quantization, again
------------------------
Same trade as `opqad/integrations/kivi_vllm_fakequant.py`: quantize-dequantize inside the
BF16 cache.  The values handed to the attention kernel are bit-identical to what
a packed cache would hand it, so the arm is exact on accuracy and silent on
memory and throughput.  Those are the axes a real paged layout would have to
earn, and the point of measuring accuracy first is to find out whether that
kernel project is worth starting.

FLASH_ATTN only, and why
------------------------
The window needs each request's *ordered block list*, which is
`attn_metadata.block_table` on FlashAttention.  FlashInfer keeps the same
information split across `prefill`/`decode` sub-objects whose shape differs
between the FI and TRTLLM variants.  Rather than guess across four layouts --
the mistake that made the first KIVI arm measure plain BF16 -- this refuses any
metadata that does not carry `block_table` and `seq_lens` directly, and the
launcher pins ``VLLM_ATTENTION_BACKEND=FLASH_ATTN``.

Eager only, for the same reason the KIVI arm is: the eligible-block set is
recomputed in Python every step, and a captured graph would replay one step's
set forever.
"""
from __future__ import annotations

import os
import re

from opqad.utils.envreg import env_float, env_int, env_optint, env_str

_applied = False


def enabled() -> bool:
    return os.environ.get("QAT_HYBRID_FAKEQUANT", "").strip() == "1"


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def layer_index(layer) -> int:
    """The decoder layer this attention module belongs to.

    The rotation is per layer -- `_layer_rotation`'s docstring records that
    applying layer 0's matrix everywhere is not a milder OSCAR but a different
    and worse transform -- so a wrong or missing index is a silent accuracy bug.
    vLLM names the module `model.layers.<i>.self_attn.attn`; if that ever stops
    being true this raises rather than defaulting to 0.
    """
    name = getattr(layer, "layer_name", None) or ""
    m = _LAYER_RE.search(name)
    if not m:
        raise ValueError(
            f"cannot read a layer index out of layer_name {name!r}; the V "
            "rotation is per layer and guessing one would quantize every layer "
            "in the wrong basis"
        )
    return int(m.group(1))


def eligible_blocks(block_table, seq_lens, block_size: int,
                    sink_tokens: int, recent_tokens: int):
    """Split each request's blocks into (quantizable, protected).

    A block is quantizable when it is entirely behind the recent window and
    entirely past the sink -- both boundaries are rounded *outward* so the
    protected regions are never smaller than asked for.  A block that is
    partially inside a protected region stays BF16: rounding the other way
    would quantize part of the window while the log said the window was 256.

    Returns ``(quantizable_ids, protected_ids)``.  Protected ids matter for
    recycling, not for quantization -- see `maybe_apply`.
    """
    import torch

    n = block_table.shape[0]
    n = min(n, seq_lens.shape[0])
    if n <= 0:
        return None, None
    bt = block_table[:n]
    seq = seq_lens[:n].to(torch.int64)

    held = torch.div(seq + block_size - 1, block_size, rounding_mode="floor")
    # First quantizable column: the sink ends mid-block, so round up.
    first = (sink_tokens + block_size - 1) // block_size
    # One past the last quantizable column: a block is behind the window only
    # if its last token is, so round the window up too.
    recent_blocks = (recent_tokens + block_size - 1) // block_size
    last = held - recent_blocks

    cols = torch.arange(bt.shape[1], device=bt.device).view(1, -1)
    live = cols < held.view(-1, 1)
    quant = live & (cols >= first) & (cols < last.view(-1, 1))
    return bt[quant].to(torch.int64), bt[live & ~quant].to(torch.int64)


def newly_quantizable(block_table, seq_lens, block_size: int, sink_tokens: int,
                      recent_tokens: int, done):
    """Blocks to quantize on this step, updating the ``done`` ledger in place.

    ``done`` is a bool tensor indexed by block id.  It exists because OSCAR's
    value rule is **not** idempotent -- the percentile clip re-clips an already
    quantized block, so quantizing one twice is real extra damage.  (KIVI's rule
    is idempotent, which is why the KIVI hook's bookkeeping could be an
    optimization and this one cannot.)

    The ledger is cleared for every *protected* block each step.  That is what
    makes block recycling safe without tracking request identity: a block always
    enters a request as that request's newest block, hence inside the recent
    window, hence protected -- so a recycled id is cleared before it can ever
    become quantizable again.  The invariant needs ``recent_tokens >= 1`` and no
    block sharing; `enable_prefix_caching` is False in the eval path.
    """
    quant, prot = eligible_blocks(block_table, seq_lens, block_size,
                                  sink_tokens, recent_tokens)
    if quant is None:
        return None
    if prot.numel():
        done.index_fill_(0, prot.unique(), False)
    if quant.numel() == 0:
        return quant
    quant = quant.unique()
    new = quant[~done.index_select(0, quant)]
    if new.numel():
        done.index_fill_(0, new, True)
    return new


def quantize_k_per_channel(kc, ids, token_axis: int, levels: int, dtype):
    """KIVI's key rule: one scale per channel, shared across the block's tokens."""
    import torch

    k = kc.index_select(0, ids).float()
    lo = k.amin(dim=token_axis, keepdim=True)
    hi = k.amax(dim=token_axis, keepdim=True)
    scale = (hi - lo) / levels
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = ((k - lo) / safe).clamp(0, levels).round() * safe + lo
    kc.index_copy_(0, ids, q.to(dtype))


def quantize_oscar_axis(cache, ids, rot, levels: int, group_size: int,
                        clip_ratio: float, dtype):
    """OSCAR's rule: rotate, per-vector quantize over head_dim, un-rotate.

    head_dim is the last axis in every paged layout vLLM uses, and this rule
    reduces over exactly that axis, so unlike the per-channel key rule it needs
    no layout branch -- there is no token axis in the reduction to get wrong.

    `_quant_dequant` is imported rather than transcribed: the clip arithmetic
    has already been got wrong once in this repo (signed two-sided quantiles vs
    serving's symmetric magnitude threshold), and one copy is the only way the
    arm and the training quantizer cannot drift.
    """
    from opqad.quantization.oscar_fake_quant import _quant_dequant

    x = cache.index_select(0, ids).float()
    if rot is not None:
        r = rot.to(device=x.device, dtype=x.dtype)
        y = x @ r
    else:
        y = x
    q = _quant_dequant(y, levels=levels, group_size=group_size,
                       clip_ratio=clip_ratio)
    if rot is not None:
        q = q @ r.transpose(-1, -2)
    cache.index_copy_(0, ids, q.to(dtype))


def maybe_apply(verbose: bool = True) -> bool:
    global _applied
    if _applied or not enabled():
        return False

    try:
        import importlib

        import torch
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[hybrid] not applied: {type(exc).__name__}: {exc}", flush=True)
        return False

    from opqad.integrations.kivi_vllm_fakequant import kv_views
    from opqad.quantization.oscar_fake_quant import _layer_rotation, _load_rotation

    bits = env_int("QAT_HYBRID_BITS")
    k_scheme = env_str("QAT_HYBRID_K_SCHEME")
    v_scheme = env_str("QAT_HYBRID_V_SCHEME")
    sink = env_int("QAT_HYBRID_SINK_TOKENS")
    recent = env_int("QAT_HYBRID_RECENT_TOKENS")
    if k_scheme not in ("channel", "oscar"):
        raise ValueError(f"QAT_HYBRID_K_SCHEME must be channel|oscar, got {k_scheme}")
    if v_scheme not in ("oscar", "kivi"):
        raise ValueError(f"QAT_HYBRID_V_SCHEME must be oscar|kivi, got {v_scheme}")
    if recent <= 0:
        # A recycled block is always the newest block of its new owner, so it
        # passes through the protected set before it can become quantizable --
        # which is what lets the `done` ledger below be cleared safely.  With no
        # recent window that invariant is gone and a recycled block could keep a
        # previous request's "already quantized" mark.
        raise ValueError("QAT_HYBRID_RECENT_TOKENS must be >= 1")

    # Same env vars the serving and training OSCAR paths read, so one number
    # means one thing everywhere.
    k_rot_payload = _load_rotation("VLLM_OSCAR_K_ROTATION_PATH")
    v_rot_payload = _load_rotation("VLLM_OSCAR_V_ROTATION_PATH")
    # The clip ratios are REQUIRED in opqad/utils/envreg.py.  They were spelled
    # 0.96/0.92 here and 0.0/0.0 in oscar_fake_quant.py, so "the same env vars
    # ... one number means one thing everywhere" was not true when neither was
    # set.  Every launcher that runs this hook sets both.
    k_clip = env_float("VLLM_OSCAR_K_CLIP_RATIO")
    v_clip = env_float("VLLM_OSCAR_V_CLIP_RATIO")
    group_size = env_optint("VLLM_OSCAR_GROUP_SIZE")
    if group_size is None:
        group_size = 128
    # This hook has always quantized every layer, which is what made it a clean
    # control against the serving path's skip list.  Since 2026-09-17 that is
    # not a default but the only behaviour -- the boundary skip is a property
    # of the upstream vLLM port, not an experimental variable.  The call raises
    # on a stale VLLM_OSCAR_QUANT_ALL_LAYERS=0 rather than honouring it.
    #
    # From the module that defines it, not from turboquant_attn_hook, which
    # re-exports it and drags in the whole native hook on import.
    from opqad.quantization.turboquant_fake_quant import oscar_quant_all_layers
    oscar_quant_all_layers()
    skip_layers: frozenset[int] = frozenset()
    if v_scheme == "oscar" and v_rot_payload is None:
        raise ValueError(
            "QAT_HYBRID_V_SCHEME=oscar but VLLM_OSCAR_V_ROTATION_PATH is unset "
            "or did not load. An unset rotation is silently the identity, which "
            "is a different and worse quantizer wearing OSCAR's name."
        )
    if k_scheme == "oscar" and k_rot_payload is None:
        raise ValueError("QAT_HYBRID_K_SCHEME=oscar but no K rotation loaded")

    step = {"key": None, "quant": None, "prot": None}
    state = {"calls": 0, "blocks": 0, "logged": False, "done": None}

    def _step_sets(attn_metadata, block_size):
        key = id(attn_metadata)
        if step["key"] == key:
            return step["quant"]
        bt = getattr(attn_metadata, "block_table", None)
        seq = getattr(attn_metadata, "seq_lens", None)
        if bt is None or seq is None:
            raise RuntimeError(
                "[hybrid] this attention metadata carries no block_table/"
                f"seq_lens ({type(attn_metadata).__name__}); the BF16 window "
                "needs each request's ordered block list. Pin "
                "VLLM_ATTENTION_BACKEND=FLASH_ATTN for this arm."
            )
        new = newly_quantizable(bt, seq, block_size, sink, recent,
                                state["done"])
        step["key"], step["quant"] = key, new
        return new

    def _wrap(original, backend: str):
        def _forward(self, layer, query, key, value, kv_cache, attn_metadata,
                     output, *a, **kw):
            out = original(self, layer, query, key, value, kv_cache,
                           attn_metadata, output, *a, **kw)
            if attn_metadata is None or kv_cache is None or kv_cache.numel() == 0:
                return out
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "[hybrid] the fake quantizer is being CUDA-graph captured. "
                    "Run this arm eager (EVAL_ENFORCE_EAGER=1): it recomputes "
                    "the eligible-block set every step, and a captured graph "
                    "would replay one step's set for the whole run."
                )
            state["calls"] += 1
            try:
                hk, d = key.shape[-2], key.shape[-1]
                kc, vc, tax, block_size = kv_views(kv_cache, hk, d)
                if state["done"] is None:
                    state["done"] = torch.zeros(kc.shape[0], dtype=torch.bool,
                                                device=kc.device)
                ids = _step_sets(attn_metadata, block_size)
                if ids is None or ids.numel() == 0:
                    return out
                idx = layer_index(layer)
                if idx in skip_layers:
                    # Reproduce upstream's boundary protection on the fake-quant
                    # path, which otherwise has no skip list at all.  Needed as a
                    # control: on Qwen2-1.5B, quantizing all 28 layers stops the
                    # model emitting EOS (8k smoke 62871: eos_fraction 0.0,
                    # truncated_fraction 1.0), and this is the knob that decides
                    # whether that is OSCAR's algorithm or our serving path.
                    return out
                levels = (1 << bits) - 1

                if k_scheme == "channel":
                    quantize_k_per_channel(kc, ids, tax, levels, kv_cache.dtype)
                else:
                    quantize_oscar_axis(
                        kc, ids, _layer_rotation(k_rot_payload, idx, d),
                        1 << bits, group_size, k_clip, kv_cache.dtype)

                if v_scheme == "oscar":
                    quantize_oscar_axis(
                        vc, ids, _layer_rotation(v_rot_payload, idx, d),
                        1 << bits, group_size, v_clip, kv_cache.dtype)
                else:
                    v = vc.index_select(0, ids).float()
                    vg = env_int("QAT_KIVI_VALUE_GROUP")
                    g = v.reshape(*v.shape[:-1], d // vg, vg)
                    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
                    scale = (hi - lo) / levels
                    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
                    q = ((g - lo) / safe).clamp(0, levels).round() * safe + lo
                    vc.index_copy_(0, ids,
                                   q.reshape(v.shape).to(kv_cache.dtype))

                state["blocks"] += int(ids.numel())
                if not state["logged"]:
                    state["logged"] = True
                    # The skip list belongs HERE, not in maybe_apply's banner:
                    # vllm_plugin calls maybe_apply(verbose=False), so that
                    # banner never prints in a real eval, and job 63442 ran the
                    # skip-4 control with nothing in its log to prove it.  This
                    # line fires from the first batch that actually quantizes,
                    # so it proves the hook ran AND which layers it spares.
                    lay = ("all layers" if not skip_layers
                           else f"skipping layers {sorted(skip_layers)}")
                    print(f"[hybrid] live on {backend}: bits={bits}, "
                          f"K={k_scheme} (token axis {tax}, block "
                          f"{block_size}), V={v_scheme} (clip {v_clip}, group "
                          f"{group_size}), BF16 sink {sink} + recent {recent} "
                          f"tokens per request, {lay}", flush=True)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"[hybrid] fake quant failed on a live batch ({backend}, "
                    f"kv_cache {tuple(kv_cache.shape)}, key {tuple(key.shape)}): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if state["blocks"] == 0 and state["calls"] > 8192:
                # The window means nothing is quantized until a request is
                # sink+recent+1 blocks long, so this threshold is higher than
                # the KIVI arm's -- but by 8192 calls at 32 seats a run at 32k
                # has long since passed it, and zero blocks can only mean the
                # wrong thing is wrapped.
                raise RuntimeError(
                    f"[hybrid] {state['calls']} attention calls on {backend} "
                    "and not one block quantized. The arm is running "
                    "unquantized BF16; refusing to report it as a 2-bit arm."
                )
            return out

        return _forward

    patched = []
    for mod_name, cls_name in (
        ("vllm.v1.attention.backends.flash_attn", "FlashAttentionImpl"),
        ("vllm.v1.attention.backends.flashinfer", "FlashInferImpl"),
        ("vllm.v1.attention.backends.triton_attn", "TritonAttentionImpl"),
    ):
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
        except Exception:  # noqa: BLE001, S112
            continue
        cls.forward = _wrap(cls.forward, cls_name)
        patched.append(cls_name)
    if not patched:
        raise RuntimeError("[hybrid] no v1 attention backend could be wrapped")

    _applied = True
    if verbose:
        _lay = (f"all layers" if not skip_layers
                else f"skipping {sorted(skip_layers)}")
        print(f"[hybrid] hybrid-{bits}bit fake quantization ON "
              f"(K={k_scheme}, V={v_scheme}, sink={sink}, recent={recent}, "
              f"{_lay}), wrapping {', '.join(patched)}. Accuracy-faithful; the "
              f"cache stays BF16, so memory and throughput are NOT this "
              f"scheme's.", flush=True)
    return True
