"""Align a quantized student's K/V to a full-precision teacher's K/V.

Why KV instead of (or in addition to) hidden states
---------------------------------------------------
Hidden-state alignment anchors the post-block representation; KV-cache
alignment anchors the *cache itself*.  For KV-cache QAT the whole point is the
cache: quantization error is injected at every block's K/V, and a hidden-state
term observes it only after 28 layers of mixing, while a logits term observes it
after the LM head.  Aligning the quantized K/V that is actually written to the
cache to the K/V the teacher would write is the most direct signal available.

The interception point is the same one the quantizer already uses
(``turboquant_attn_hook``): post-RoPE, un-repeated, ``head_dim`` last.  The
student branch is the dequantized K/V the wrapper feeds to SDPA -- i.e. the
quantizer *output*, which is differentiable.  The teacher branch is the same
weights with the quantizer bypassed, run either as the frozen theta_0 copy
(``frozen_model``) or, when that is unavailable, as the current weights under
``state.capture_only()``.

The frozen theta_0 copy matters for the same reason it does in
``block_align_qat``: a co-moving reference is first-order invariant under the
straight-through estimator, so the distance carries no gradient to close.  Use
``frozen_model`` whenever one is available.  The frozen reference does not
carry the quantizer hook by construction (it is full precision); this module
installs a *capture-only* attention hook on it so its raw post-RoPE K/V appear
at the identical interception point as the student's.

Metric choice
-------------
K/V activations are not monotone across depth the way hidden states are, but
their magnitude still varies and, critically, key vectors are normalized per
token by the quantizer so a scale-free metric keeps the term well-posed.  Both
metrics below are scale-free per (position, head):

  ``cosine``  1 - cos(k_s, k_t)
  ``nmse``    ||k_s - k_t||^2 / (||k_t||^2 + eps)

``mse`` is offered for the value stream, whose per-channel min-max is bounded
and where a raw squared error is a reasonable objective, but cosine is the
safest default for keys.

Everything is reduced in FP32 regardless of input dtype.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .block_align import subsample_positions

__all__ = [
    "KVAlignConfig",
    "compute_kv_alignment",
    "reference_kv_states",
]

_EPS = 1e-6


class KVAlignConfig:
    """Knobs for the K/V alignment term.

    Args:
        layer_every: align every N-th quantized layer (plus the last).  The KV
            tensors are per-layer and must survive until backward.
        keep_fraction: fraction of response positions to score.
        metric: ``"nmse"``, ``"cosine"``, or ``"mse"`` (scalar per-position, or
            per-token-per-head for the first two).
        coef: weight of the term in the total loss.
        align_keys / align_values: whether the key and/or value stream is
            scored.  Both default to True.
    """

    def __init__(
        self,
        layer_every: int = 1,
        keep_fraction: float = 1.0,
        metric: str = "nmse",
        coef: float = 1.0,
        align_keys: bool = True,
        align_values: bool = True,
    ):
        if coef < 0:
            raise ValueError(f"coef must be non-negative, got {coef}")
        if metric not in ("nmse", "cosine", "mse"):
            raise ValueError(f"unknown metric {metric!r}, expected nmse/cosine/mse")
        if not (align_keys or align_values):
            raise ValueError("at least one of align_keys/align_values must be True")
        self.layer_every = layer_every
        self.keep_fraction = keep_fraction
        self.metric = metric
        self.coef = coef
        self.align_keys = align_keys
        self.align_values = align_values

    def metadata(self) -> dict:
        return {
            "kv_align/layer_every": self.layer_every,
            "kv_align/keep_fraction": self.keep_fraction,
            "kv_align/metric": self.metric,
            "kv_align/coef": self.coef,
            "kv_align/align_keys": float(self.align_keys),
            "kv_align/align_values": float(self.align_values),
        }


def _capture_only_reference(model, preset):
    """Install a capture-only attention hook on ``model``, idempotently.

    Returns the state.  A capture-only state routes attention through the same
    wrapper the quantizer uses but bypasses quantization and records the raw
    post-RoPE K/V, so the frozen teacher's K/V appear at the exact interception
    point of the student's.
    """
    from .turboquant_attn_hook import (
        get_installed_state,
        install_turboquant_fake_quant,
    )

    existing = get_installed_state(model)
    if existing is not None:
        return existing
    return install_turboquant_fake_quant(model, preset, capture_only=True)


def _prep_kv(t, keep_heads: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Return ``t`` as ``[B, H, T, D]`` float, optionally filtered by head."""
    # The attention interface hands (batch, num_kv_heads, seq, head_dim).
    if t.ndim == 5:  # pragma: no cover - defensive
        t = t.squeeze(0)
    if t.ndim == 3:
        t = t.unsqueeze(1)
    if keep_heads is not None:
        t = t[:, keep_heads]
    return t


@torch.no_grad()
def reference_kv_states(
    model,
    state,
    model_inputs: Dict[str, Any],
    *,
    preset,
    autocast_dtype: Optional[torch.dtype] = None,
    device_type: str = "cuda",
    frozen_model=None,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Full-precision K/V to align the quantized forward against.

    ``frozen_model`` is the theta_0 copy from ``fsdp_workers``; it yields a
    stationary target.  When it is None the current weights are used with the
    quantizer bypassed (co-moving target; degenerate under STE -- only for
    fallback, prefer ``frozen_model``).

    Returns a dict ``{layer_idx: (key, value)}`` for the quantized layer set,
    each a detached ``[B, H, T, D]`` float tensor.
    """
    from .turboquant_attn_hook import get_installed_state

    if frozen_model is not None:
        ref_state = _capture_only_reference(frozen_model, preset)
        with contextlib.ExitStack() as stack:
            if autocast_dtype is not None:
                stack.enter_context(
                    torch.autocast(device_type=device_type, dtype=autocast_dtype)
                )
            with ref_state.capture_only():
                frozen_model(**model_inputs, output_hidden_states=False, use_cache=False)
        captured = ref_state.kv
    else:
        was_training = model.training
        model.eval()
        try:
            with contextlib.ExitStack() as stack:
                if autocast_dtype is not None:
                    stack.enter_context(
                        torch.autocast(device_type=device_type, dtype=autocast_dtype)
                    )
                with state.capture_only():
                    model(**model_inputs, output_hidden_states=False, use_cache=False)
        finally:
            if was_training:
                model.train()
        captured = state.kv

    out = {}
    for layer_idx, (k, v) in captured.items():
        out[int(layer_idx)] = (
            _prep_kv(k.clone().detach()),
            _prep_kv(v.clone().detach()),
        )
    return out


def _distance(s: torch.Tensor, t: torch.Tensor, metric: str) -> torch.Tensor:
    """Per-(position, head) distance, ``[B, H, T]``."""
    if metric == "cosine":
        return 1.0 - torch.nn.functional.cosine_similarity(s, t, dim=-1, eps=_EPS)
    if metric == "mse":
        return (s - t).pow(2).mean(dim=-1)
    num = (s - t).pow(2).sum(dim=-1)
    den = t.pow(2).sum(dim=-1) + _EPS
    return num / den


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` ``[B, H, T]`` over ``mask`` ``[B, T]`` broadcast."""
    mask3 = mask.to(torch.float32).unsqueeze(1)  # [B,1,T]
    denom = mask3.sum()
    if denom == 0:
        return values.sum() * 0.0
    return (values * mask3).sum() / denom


def compute_kv_alignment(
    model,
    state,
    model_inputs: Dict[str, Any],
    response_mask: torch.Tensor,
    cfg: KVAlignConfig,
    *,
    student_kv: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    generator: Optional[torch.Generator] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    device_type: str = "cuda",
    frozen_model=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Alignment loss between the student's quantized K/V and the teacher's.

    Args:
        model / state: the actor and its installed ``TurboQuantAttnState``.
        model_inputs: kwargs for ``model(**model_inputs)``.
        response_mask: ``[B, T]`` over the positions to score.
        cfg: :class:`KVAlignConfig`.
        student_kv: ``{layer_idx: (key, value)}`` from the quantized forward,
            differentiable, if the caller already captured it (``update_policy``
            does via ``state.capturing_kv()``).  When omitted a second
            quantized forward is run to capture it.
        frozen_model: theta_0 reference, or None for the co-moving fallback.

    Returns:
        ``(loss, metrics)``.  ``loss`` already scaled by ``cfg.coef``.
    """
    if not state.enabled and frozen_model is None:
        raise RuntimeError("the quantizer is disabled; there is nothing to align")

    from .turboquant_attn_hook import get_installed_state

    teacher = reference_kv_states(
        model, state, model_inputs,
        preset=state.cfg, autocast_dtype=autocast_dtype,
        device_type=device_type, frozen_model=frozen_model,
    )

    if student_kv is None:
        was_training = model.training
        model.train()
        try:
            with contextlib.ExitStack() as stack:
                if autocast_dtype is not None:
                    stack.enter_context(
                        torch.autocast(device_type=device_type, dtype=autocast_dtype)
                    )
                with state.capturing_kv() as kv:
                    model(**model_inputs, output_hidden_states=False, use_cache=False)
        finally:
            if not was_training:
                model.eval()
        student_kv = kv

    mask = subsample_positions(response_mask, cfg.keep_fraction, generator=generator)

    # Layers to score: the layers the hook actually quantizes (skip-layers
    # excluded), subsampled every ``layer_every``, always keeping the last
    # quantized layer so the deepest quantizer sees the term.
    quantized = sorted(set(teacher.keys()) - set(state.skip_layers))
    if not quantized:
        raise RuntimeError("no quantized layers captured by either forward")
    chosen = list(range(0, len(quantized), cfg.layer_every))
    if (len(quantized) - 1) not in chosen:
        chosen.append(len(quantized) - 1)
    chosen = [quantized[i] for i in sorted(chosen)]

    loss = None
    metrics: Dict[str, float] = {}
    weighted = len(chosen)
    for layer_idx in chosen:
        s_key, s_val = student_kv[layer_idx]
        t_key, t_val = teacher[layer_idx]
        if s_key.shape != t_key.shape:
            raise ValueError(
                f"layer {layer_idx}: student key {tuple(s_key.shape)} != "
                f"teacher key {tuple(t_key.shape)}"
            )
        if s_key.shape[0] != mask.shape[0] or s_key.shape[2] != mask.shape[1]:
            raise ValueError(
                f"layer {layer_idx}: KV {tuple(s_key.shape[:3])} does not match "
                f"mask {tuple(mask.shape)}"
            )
        if cfg.align_keys:
            dk = _distance(s_key.float(), t_key.float().detach(), cfg.metric)
            mk = _masked_mean(dk, mask)
            metrics[f"kv_align/layer_{layer_idx}_key"] = mk.detach().item()
            loss = mk / weighted if loss is None else loss + mk / weighted
        if cfg.align_values:
            dv = _distance(s_val.float(), t_val.float().detach(), cfg.metric)
            mv = _masked_mean(dv, mask)
            metrics[f"kv_align/layer_{layer_idx}_value"] = mv.detach().item()
            loss = mv / weighted if loss is None else loss + mv / weighted

    assert loss is not None
    metrics["kv_align/loss"] = loss.detach().item()
    metrics["kv_align/tokens"] = float(mask.sum().item())
    metrics["kv_align/layers"] = float(len(chosen))
    metrics.update(cfg.metadata())
    return loss * cfg.coef, metrics
