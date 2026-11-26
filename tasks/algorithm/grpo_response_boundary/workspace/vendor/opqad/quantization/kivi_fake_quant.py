"""KIVI KV-cache fake quantization adapter.

The quantization rules below are adapted from the official KIVI repository
(MIT, commit 876b4d2), ``quant/new_pack.py``.  We deliberately keep KIVI's
asymmetric min/max rules and only replace its packed CUDA storage with
quantize-then-dequantize tensors so the Qwen2.5 evaluator can measure the
numerical effect without pretending to have an integer cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class KIVIFakeQuantConfig:
    bits: int = 2
    group_size: int = 32
    residual_length: int = 32

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or self.bits not in (2, 4, 8):
            raise ValueError("KIVI fake quant supports 2, 4, or 8 bits")
        if isinstance(self.group_size, bool) or self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if isinstance(self.residual_length, bool) or self.residual_length < 0:
            raise ValueError("residual_length must be non-negative")
        # residual_length=0 removes KIVI's high-precision window entirely, so
        # every position goes through the quantiser.  KIVI itself never runs
        # this way; it is here because the window is hostile to training --
        # which tokens are exempt depends on the current position, so a QAT
        # student can learn to lean on "the recent tokens are clean", a crutch
        # that does not survive deployment.  With the window gone the quantiser
        # is a plain static function of the tensor and what QAT learns
        # transfers.
        if self.residual_length and self.residual_length % self.group_size:
            raise ValueError("KIVI requires residual_length divisible by group_size")


def _quantize_kcache(k: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    """KIVI's key rule: asymmetric min/max over token groups per channel."""
    if k.ndim != 4 or k.shape[-2] % group_size:
        raise ValueError("KIVI key cache requires [B,H,T,D] and T divisible by group_size")
    bsz, heads, tokens, dim = k.shape
    grouped = k.reshape(bsz, heads, tokens // group_size, group_size, dim)
    max_int = 2**bits - 1
    mn = grouped.amin(dim=-2, keepdim=True)
    mx = grouped.amax(dim=-2, keepdim=True)
    scale = (mx - mn) / max_int
    # KIVI's CUDA kernel receives the same min/max metadata.  A constant
    # group has no representable range; preserve its exact value instead of
    # allowing a zero scale to create NaNs in a fake implementation.
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    code = ((grouped - mn) / safe_scale).clamp(0, max_int).round()
    restored = code * safe_scale + mn
    return restored.reshape_as(k).to(dtype=k.dtype)


def _quantize_vcache(v: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    """KIVI's value rule: asymmetric min/max over channel groups per token."""
    if v.ndim != 4 or v.shape[-1] % group_size:
        raise ValueError("KIVI value cache requires [B,H,T,D] and D divisible by group_size")
    bsz, heads, tokens, dim = v.shape
    grouped = v.reshape(bsz, heads, tokens, dim // group_size, group_size)
    max_int = 2**bits - 1
    mn = grouped.amin(dim=-1, keepdim=True)
    mx = grouped.amax(dim=-1, keepdim=True)
    scale = (mx - mn) / max_int
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    code = ((grouped - mn) / safe_scale).clamp(0, max_int).round()
    restored = code * safe_scale + mn
    return restored.reshape_as(v).to(dtype=v.dtype)


def fake_quantize_cache(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    config: KIVIFakeQuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply KIVI quantize/dequantize while retaining its residual window."""
    if key_states.shape != value_states.shape:
        raise ValueError("KIVI key/value cache shapes must match")
    tokens = key_states.shape[-2]
    # This follows KIVI's two different residual rules in llama_kivi.py:
    # keys keep only the remainder needed for residual_length alignment,
    # while values always retain exactly the most recent residual_length.
    if config.residual_length == 0:
        # No window: quantise everything.  Keys still need a whole number of
        # groups, so a ragged tail shorter than group_size stays in full
        # precision -- that is a shape constraint of the grouping, not a
        # residual window, and it vanishes once T >= group_size.
        key_quant_tokens = (tokens // config.group_size) * config.group_size
        value_quant_tokens = tokens
    else:
        key_quant_tokens = (tokens // config.residual_length) * config.residual_length
        value_quant_tokens = max(0, tokens - config.residual_length)
    key_prefix = (
        _quantize_kcache(key_states[..., :key_quant_tokens, :], config.group_size, config.bits)
        if key_quant_tokens
        else key_states[..., :0, :]
    )
    value_prefix = (
        _quantize_vcache(value_states[..., :value_quant_tokens, :], config.group_size, config.bits)
        if value_quant_tokens
        else value_states[..., :0, :]
    )
    return (
        torch.cat((key_prefix, key_states[..., key_quant_tokens:, :]), dim=-2),
        torch.cat((value_prefix, value_states[..., value_quant_tokens:, :]), dim=-2),
    )


class KIVICacheFakeQuantHandle:
    """Temporarily patch Transformers' dynamic cache update method."""

    def __init__(self, config: KIVIFakeQuantConfig):
        self.config = config
        self._original: Any = None
        self.hits = 0
        # Progress is tracked per cache layer (see ``update``), not here.

    def install(self) -> "KIVICacheFakeQuantHandle":
        from transformers.cache_utils import Cache

        if self._original is not None:
            return self
        self._original = Cache.update
        handle = self

        def update(cache: Any, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
            keys, values = handle._original(cache, key_states, value_states, layer_idx, cache_kwargs)
            layer = cache.layers[layer_idx]
            # KIVI's persistent representation is a packed side cache.  This
            # fake path writes the dequantized tensor back into DynamicLayer;
            # other cache implementations are rejected instead of silently
            # applying a different algorithm.
            if layer.__class__.__name__ not in {"DynamicLayer", "DynamicSlidingWindowLayer"}:
                raise RuntimeError("KIVI fake quant currently requires Transformers DynamicCache")

            # Quantise only what has newly become eligible, then splice.
            #
            # The previous version re-quantised the whole cache on every token,
            # which is O(n^2) over a generation: at 7.5k tokens an AIME answer
            # spent hours making no progress (job 8928, cancelled after 3h with
            # both GPUs at 100% CPU and an empty log).  Worse, it wrote the
            # result back into the layer, so each pass re-quantised
            # already-quantised values and compounded the error.
            #
            # Because the result is written back, everything before
            # ``done`` is already in its final quantised form and must be left
            # alone.
            cfg = handle.config
            tokens = keys.shape[-2]
            # The progress counters live on the layer, not on the handle.
            # The handle is installed once and outlives every generation, so
            # a handle-keyed counter carried problem N's progress into
            # problem N+1: the next sample starts a fresh cache at
            # T=prompt_len, far below the retained counter, so nothing ever
            # looked "newly eligible" and every generation after the first
            # ran unquantised.  Transformers builds a fresh DynamicCache --
            # and fresh layer objects -- per generate(), so layer-attached
            # state resets exactly when the sequence does.
            done = getattr(layer, "_kivi_done", 0)
            if tokens < done:  # cache shrank: a new sequence reused this layer
                done = 0

            # Keys quantise in whole groups; values keep the most recent
            # residual_length exact.  Both rules match fake_quantize_cache.
            if cfg.residual_length == 0:
                key_target = (tokens // cfg.group_size) * cfg.group_size
                value_target = tokens
            else:
                key_target = (tokens // cfg.residual_length) * cfg.residual_length
                value_target = max(0, tokens - cfg.residual_length)

            if key_target > done:
                new_k = _quantize_kcache(
                    keys[..., done:key_target, :], cfg.group_size, cfg.bits
                )
                keys = torch.cat(
                    (keys[..., :done, :], new_k, keys[..., key_target:, :]), dim=-2
                )
                layer._kivi_done = key_target

            # Values group along channels, so each token can be finalised on
            # its own -- but only once.  Quantising [0, value_target) every step
            # would keep the whole thing quadratic even after fixing the keys.
            vdone = getattr(layer, "_kivi_vdone", 0)
            if tokens < vdone:
                vdone = 0
            if value_target > vdone:
                new_v = _quantize_vcache(
                    values[..., vdone:value_target, :], cfg.group_size, cfg.bits
                )
                values = torch.cat(
                    (values[..., :vdone, :], new_v, values[..., value_target:, :]),
                    dim=-2,
                )
                layer._kivi_vdone = value_target

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


def install_kivi_fake_quant(*, bits: int = 2, group_size: int = 32, residual_length: int = 32) -> KIVICacheFakeQuantHandle:
    return KIVICacheFakeQuantHandle(KIVIFakeQuantConfig(bits, group_size, residual_length)).install()
