"""Naive per-token KV-cache fake quantization -- the LLM-QAT-style baseline.

LLM-QAT (arXiv:2305.17888) quantizes activations per token and treats the KV
cache the same way, so each (token, head) vector is quantized with its own
asymmetric min/max over ``head_dim``.  That rule is also the baseline KIVI
measures itself against, which is why it is the right reference point here:
"we beat the naive baseline" is only meaningful if the naive baseline is the
one the prior work used.

The paper's exact granularity could not be confirmed from the PDF, the arXiv
abstract or the public repo at the time of writing, so ``group_size`` is
explicit rather than assumed: the default (``None``) is one scale per token
per head, and a smaller value subdivides ``head_dim`` for a strictly stronger
baseline.  Both are reported rather than silently picking the flattering one.

Unlike KIVI this rule is token-local -- a token can be quantized the moment it
arrives, with no group to fill and no residual window -- which is also why it
maps onto a paged KV cache without a staging buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "NaiveKVFakeQuantConfig",
    "naive_per_token_quantize",
    "install_naive_kv_fake_quant",
]


@dataclass(frozen=True)
class NaiveKVFakeQuantConfig:
    bits: int = 4
    # None -> one scale per (token, head) over the whole head_dim.
    group_size: int | None = None
    symmetric: bool = False
    # Keep the most recent N *values* in full precision.  Measured on KIVI:
    # the windowed and windowless variants quantise keys byte-identically
    # (max|dK| = 0), so the entire 41.67 vs 28.33 gap on AIME24 comes from this
    # value tail alone.  It is separated from the key rule here so the two can
    # be credited independently.
    value_residual_length: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or self.bits not in (2, 3, 4, 8):
            raise ValueError(f"naive KV quant supports 2, 3, 4 or 8 bits, got {self.bits}")
        if self.group_size is not None and (
            isinstance(self.group_size, bool) or self.group_size <= 0
        ):
            raise ValueError("group_size must be positive or None")


def naive_per_token_quantize(x: torch.Tensor, cfg: NaiveKVFakeQuantConfig) -> torch.Tensor:
    """Quantize/dequantize ``[B, H, T, D]`` independently for each token."""
    if x.ndim != 4:
        raise ValueError("naive KV quant requires [B, H, T, D]")
    b, h, t, d = x.shape
    g = cfg.group_size or d
    if d % g:
        raise ValueError(f"head_dim {d} is not divisible by group_size {g}")
    grouped = x.reshape(b, h, t, d // g, g)
    max_int = 2**cfg.bits - 1
    if cfg.symmetric:
        absmax = grouped.abs().amax(dim=-1, keepdim=True)
        scale = absmax / (max_int // 2)
        safe = torch.where(scale == 0, torch.ones_like(scale), scale)
        code = (grouped / safe).round().clamp(-(max_int // 2) - 1, max_int // 2)
        out = code * safe
    else:
        mn = grouped.amin(dim=-1, keepdim=True)
        mx = grouped.amax(dim=-1, keepdim=True)
        scale = (mx - mn) / max_int
        # A constant group has no representable range; keep its exact value
        # instead of letting a zero scale produce NaNs.
        safe = torch.where(scale == 0, torch.ones_like(scale), scale)
        code = ((grouped - mn) / safe).round().clamp(0, max_int)
        out = code * safe + mn
    return out.reshape_as(x).to(dtype=x.dtype)


class NaiveKVFakeQuantHandle:
    """Patch Transformers' cache update so each new token is quantized once."""

    def __init__(self, config: NaiveKVFakeQuantConfig):
        self.config = config
        self._original: Any = None
        self.hits = 0

    def install(self) -> "NaiveKVFakeQuantHandle":
        from transformers.cache_utils import Cache

        if self._original is not None:
            return self
        self._original = Cache.update
        handle = self

        def update(cache, key_states, value_states, layer_idx, cache_kwargs=None):
            keys, values = handle._original(cache, key_states, value_states, layer_idx, cache_kwargs)
            layer = cache.layers[layer_idx]
            if layer.__class__.__name__ not in {"DynamicLayer", "DynamicSlidingWindowLayer"}:
                raise RuntimeError("naive KV fake quant requires Transformers DynamicCache")

            # Per-token quantisation is token-local, so only the newly arrived
            # positions need work.  The mark lives on the layer, not on this
            # handle: the handle outlives every generation, and a handle-keyed
            # counter would carry one sequence's progress into the next and
            # leave everything after the first sample unquantised.
            done = getattr(layer, "_naive_done", 0)
            tokens = keys.shape[-2]
            if tokens < done:      # a new sequence reused this layer object
                done = 0
            cfg = handle.config
            if tokens > done:
                keys = torch.cat(
                    (keys[..., :done, :], naive_per_token_quantize(keys[..., done:, :], cfg)), dim=-2
                )
                layer._naive_done = tokens

            # Values lag the keys by the residual window: a value is only
            # finalised once it has fallen out of the tail.
            vdone = getattr(layer, "_naive_vdone", 0)
            if tokens < vdone:
                vdone = 0
            v_target = max(0, tokens - cfg.value_residual_length)
            if v_target > vdone:
                values = torch.cat(
                    (values[..., :vdone, :],
                     naive_per_token_quantize(values[..., vdone:v_target, :], cfg),
                     values[..., v_target:, :]),
                    dim=-2,
                )
                layer._naive_vdone = v_target
            layer.keys = keys
            layer.values = values
            handle.hits += 1
            return keys, values

        Cache.update = update
        return self

    def remove(self) -> None:
        if self._original is None:
            return
        from transformers.cache_utils import Cache

        Cache.update = self._original
        self._original = None


def install_naive_kv_fake_quant(
    *, bits: int = 4, group_size: int | None = None, symmetric: bool = False,
    value_residual_length: int = 0,
) -> NaiveKVFakeQuantHandle:
    return NaiveKVFakeQuantHandle(
        NaiveKVFakeQuantConfig(bits=bits, group_size=group_size, symmetric=symmetric,
                               value_residual_length=value_residual_length)
    ).install()
