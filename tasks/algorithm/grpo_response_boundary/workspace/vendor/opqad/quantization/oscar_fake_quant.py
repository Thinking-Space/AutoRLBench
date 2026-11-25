"""OSCAR INT2 KV fake quantizer for the training side.

Mirrors the vLLM `oscar_int2` backend (PR #46774) at the same interception
point every other scheme here uses: post-RoPE, un-repeated K/V, head_dim last.

The operator is:  rotate -> group along head_dim -> asymmetric uniform
quantize to `levels` -> dequantize -> un-rotate.  The rotation is a per-layer
orthogonal matrix from OSCAR's offline spectral calibration (RotationZoo); with
no rotation supplied it degrades to identity, which is the right default for
models RotationZoo has no entry for (our 1.5B) and keeps the data path intact.

The serving-faithful recent-token window is implemented by
``oscar_windowed_attn.windowed_fake_quant_attention``: raw BF16 sink/recent
attention and INT2-history attention are evaluated with complementary masks
and merged by log-sum-exp.  OSCAR installation now requires that path; this
module supplies the rotate/clip/quantize/dequantize operator used by its INT2
branch.  Calling ``oscar_fake_quant`` directly still honors only the fixed sink
prefix, because a standalone K/V tensor has no query-relative recent window.

The clip arithmetic below is a transcription of the serving path
(`OscarAttentionImpl._rotate_clip` + `oscar_store`), not an independent
implementation of the same idea.  See `_quant_dequant`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from opqad.utils.envreg import env_float, env_int, env_optint

__all__ = [
    "OSCAR_PRESETS",
    "OscarFakeQuantConfig",
    "is_oscar_scheme",
    "oscar_fake_quant",
]

# levels=4 is INT2.  group_size/sink/recent default to the published serving
# recipe (VLLM_OSCAR_GROUP_SIZE=128, SINK=64, RECENT=256).
OSCAR_PRESETS: dict[str, dict] = {
    "oscar_int2": dict(key_levels=4, value_levels=4, group_size=128),
}


def is_oscar_scheme(scheme: Optional[str]) -> bool:
    return bool(scheme) and str(scheme) in OSCAR_PRESETS


@dataclass
class OscarFakeQuantConfig:
    head_dim: int
    scheme_name: str = "oscar_int2"
    key_levels: int = 4
    value_levels: int = 4
    group_size: int = 128
    sink_tokens: int = 64
    # Only read when QAT_OSCAR_WINDOWED_ATTN=1 (opqad/kernels/oscar_windowed_attn.py).
    # Kept on the config rather than fetched from the environment at the call
    # site so that the recorded settings manifest shows what a run actually
    # used -- the same reason sink_tokens lives here.
    recent_tokens: int = 256
    k_clip_ratio: float = 0.0
    v_clip_ratio: float = 0.0
    return_rotated: bool = False  # parity with the TQ config surface
    k_rotation: Optional[torch.Tensor] = None
    v_rotation: Optional[torch.Tensor] = None

    @classmethod
    def from_preset(cls, name: str, head_dim: int, **kw):
        if name not in OSCAR_PRESETS:
            raise ValueError(
                f"unknown OSCAR preset {name!r}; valid: {sorted(OSCAR_PRESETS)}"
            )
        # Read the same environment variables the vLLM backend reads, so the
        # training quantizer and the rollout quantizer cannot drift apart
        # through configuration.
        #
        # These go through opqad/utils/envreg.py: the window and the clip ratios are
        # REQUIRED there, because they are the quantizer's definition and not a
        # tuning knob, and because the helper this replaced returned the default
        # on a PARSE failure -- VLLM_OSCAR_SINK_TOKENS=64x silently became 64.
        # Every OSCAR launcher already sets all four explicitly.
        cfg = dict(OSCAR_PRESETS[name])
        group_size = env_optint("VLLM_OSCAR_GROUP_SIZE")
        if group_size is not None:      # unset keeps the preset's own
            cfg["group_size"] = group_size
        return cls(
            head_dim=head_dim,
            scheme_name=name,
            sink_tokens=env_int("VLLM_OSCAR_SINK_TOKENS"),
            recent_tokens=env_int("VLLM_OSCAR_RECENT_TOKENS"),
            k_clip_ratio=env_float("VLLM_OSCAR_K_CLIP_RATIO"),
            v_clip_ratio=env_float("VLLM_OSCAR_V_CLIP_RATIO"),
            k_rotation=_load_rotation("VLLM_OSCAR_K_ROTATION_PATH"),
            v_rotation=_load_rotation("VLLM_OSCAR_V_ROTATION_PATH"),
            **cfg,
            **kw,
        )


def _load_rotation(env_name: str) -> Optional[torch.Tensor]:
    """Load a RotationZoo checkpoint, or None when unset.

    RotationZoo files are dicts carrying metadata alongside the matrices; the
    per-layer tensor is picked up lazily by the caller.  Returning the raw
    object keeps this module agnostic to the exact layout, which differs
    between the qqt/sst and per-head variants.
    """
    path = os.environ.get(env_name, "").strip()
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"{env_name}={path!r} does not exist")
    return torch.load(path, map_location="cpu", weights_only=False)


def _ste(quantized: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
    """Straight-through: forward takes the quantized value, backward the identity."""
    return continuous + (quantized - continuous).detach()


def _quant_dequant(
    x: torch.Tensor, *, levels: int, group_size: int, clip_ratio: float
) -> torch.Tensor:
    """Asymmetric uniform quantize/dequantize over groups along head_dim.

    Transcribed from the serving path so that one `clip_ratio` number means one
    thing on both sides.  Serving is two steps:

        _rotate_clip:  thr = quantile(|x_rot|, c, dim=-1)
                       x_rot = clamp(x_rot, -thr, thr)
        oscar_store:   lo, hi = amin/amax of the *clamped* values

    This used to clip with two-sided SIGNED quantiles instead --
    `lo = quantile(x, 1-c)`, `hi = quantile(x, c)`, quantizing over `[lo, hi]`.
    That is a different operator at the same `c`: it discards both tails (2*(1-c)
    of the values, twice as many) and keeps an asymmetric zero point, where
    serving thresholds on magnitude and lands symmetric.  Harmless while both
    sides defaulted to c=0.0 and neither clipped at all, but the offline sweep
    puts the useful range at c=0.7-0.9, where the two differ substantially.
    """
    d = x.shape[-1]
    g = group_size if group_size > 0 else d
    if d % g:
        raise ValueError(f"head_dim {d} not divisible by group_size {g}")
    xg = x.reshape(*x.shape[:-1], d // g, g).float()

    if clip_ratio and 0.0 < clip_ratio < 1.0:
        # Percentile clip on the rotated coordinates.  amin/amax alone would
        # let a single outlier set the step size for all `g` values, which at
        # 2 bits is the whole budget.
        thr = torch.quantile(xg.abs(), clip_ratio, dim=-1, keepdim=True)
        xg = torch.clamp(xg, -thr, thr)

    lo = xg.amin(dim=-1, keepdim=True)
    hi = xg.amax(dim=-1, keepdim=True)

    scale = (hi - lo).clamp(min=1e-8) / (levels - 1)
    # floor(u + 0.5), matching oscar_store's `(x/scale + zero + 0.5).to(int32)`;
    # torch.round is half-to-even and would disagree on exact ties.
    q = torch.floor((xg - lo) / scale + 0.5)
    deq = q.clamp(0, levels - 1) * scale + lo
    return deq.reshape_as(x).to(x.dtype)


def _apply(
    x: torch.Tensor,
    rot: Optional[torch.Tensor],
    *,
    levels: int,
    group_size: int,
    clip_ratio: float,
    sink_tokens: int,
) -> torch.Tensor:
    """Rotate, quantize, un-rotate; leave the first `sink_tokens` untouched.

    `x` is [batch, heads, seq, head_dim] or [batch, seq, heads, head_dim];
    the sequence axis is located by matching against the non-head_dim dims is
    ambiguous, so the caller passes tensors in the hook's canonical layout
    (post-RoPE, head_dim last, sequence at dim -2 after the head axis).
    """
    orig = x
    if rot is not None:
        r = rot.to(device=x.device, dtype=torch.float32)
        y = x.float() @ r
    else:
        y = x.float()

    q = _quant_dequant(y, levels=levels, group_size=group_size, clip_ratio=clip_ratio)

    if rot is not None:
        q = q @ r.transpose(-1, -2)
    q = q.to(orig.dtype)

    if sink_tokens > 0 and orig.dim() >= 3:
        # Sink is a fixed prefix, so it is position-independent and can be
        # honoured exactly here (unlike the recent window).  Whether it SHOULD
        # be is a deployment question, not a training one -- see the module
        # docstring.  With the staging arena at 0% sink coverage this trains
        # the student to expect protection serving does not provide; set
        # VLLM_OSCAR_SINK_TOKENS=0 until the arena is fixed.
        seq_axis = -2
        n = orig.shape[seq_axis]
        keep = min(sink_tokens, n)
        if keep > 0:
            idx = torch.arange(n, device=orig.device)
            mask = (idx < keep).view(*([1] * (orig.dim() - 2)), n, 1)
            q = torch.where(mask, orig, q)

    return _ste(q, orig)


def oscar_fake_quant(
    key: torch.Tensor,
    value: torch.Tensor,
    cfg: OscarFakeQuantConfig,
    layer_idx: Optional[int] = None,
):
    """Quantize/dequantize K and V in place of the real INT2 cache.

    ``layer_idx`` selects the rotation: RotationZoo fits one 128x128 matrix per
    layer from that layer's own K/V covariances, so applying layer 0's matrix
    everywhere is not a milder version of OSCAR, it is a different and worse
    transform.  It is optional only so that the no-rotation path (and the unit
    tests, which construct explicit matrices) need not supply it; when a
    per-layer payload is configured and no index is given, ``_layer_rotation``
    raises rather than quietly falling back.
    """
    k = _apply(
        key,
        _layer_rotation(cfg.k_rotation, layer_idx, cfg.head_dim),
        levels=cfg.key_levels,
        group_size=cfg.group_size,
        clip_ratio=cfg.k_clip_ratio,
        sink_tokens=cfg.sink_tokens,
    )
    v = _apply(
        value,
        _layer_rotation(cfg.v_rotation, layer_idx, cfg.head_dim),
        levels=cfg.value_levels,
        group_size=cfg.group_size,
        clip_ratio=cfg.v_clip_ratio,
        sink_tokens=cfg.sink_tokens,
    )
    return k, v


# (id(payload), layer_idx) -> (payload, matrix).  The payload is kept in the
# value purely to pin it alive, so the id cannot be recycled under us.
_ROT_CACHE: dict = {}


def _layer_rotation(
    obj, layer_idx: Optional[int] = None, head_dim: Optional[int] = None
) -> Optional[torch.Tensor]:
    """Extract this layer's [head_dim, head_dim] matrix from a RotationZoo payload.

    The zoo layout, confirmed against the fitted files, is

        {"format_version": 1, "objective": "qqt_r_h_pbr",
         "source_grouping": "layer",
         "layers": {0: {"layer_id": 0, "rotation": [128,128], "eigenvalues": [128]}, ...}}

    The previous version of this function looked for a top-level "rotation" /
    "matrix" / "R" / "weight" key, then for any square 2-D tensor among the
    values, and returned None when it found neither.  Against the layout above
    it found neither -- "layers" is a dict of dicts, not a tensor -- so it
    returned None, and None means identity.  That is why exporting
    VLLM_OSCAR_K_ROTATION_PATH was necessary but not sufficient: every OSCAR
    *training* arm this repo has run quantized in the unrotated basis while the
    serving path it was being adapted to quantized in the rotated one.  Nothing
    logged it, because identity is a legitimate configuration.

    So this now raises on every path where a rotation was configured but cannot
    be resolved.  A missing rotation costs accuracy without changing a line of
    the log; a raise costs one failed job.
    """
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj  # a bare matrix: the caller means it for every layer

    if not isinstance(obj, dict):
        raise TypeError(f"unsupported OSCAR rotation payload: {type(obj)!r}")

    ck = (id(obj), layer_idx)
    hit = _ROT_CACHE.get(ck)
    if hit is not None:
        return hit[1]

    layers = obj.get("layers")
    if layers is not None:
        if layer_idx is None:
            raise ValueError(
                "OSCAR rotation payload is per-layer "
                f"(objective={obj.get('objective')!r}, {len(layers)} layers) but no "
                "layer_idx was supplied.  Falling back to identity here would "
                "silently un-rotate the whole model; pass module.layer_idx through."
            )
        li = int(layer_idx)
        entry = None
        if isinstance(layers, dict):
            # torch.save round-trips int keys, but tolerate str keys too.
            entry = layers.get(li, layers.get(str(li)))
        elif isinstance(layers, (list, tuple)) and 0 <= li < len(layers):
            entry = layers[li]
        if entry is None:
            raise KeyError(
                f"OSCAR rotation payload has no entry for layer {li} "
                f"({len(layers)} layers present)"
            )
        mat = entry["rotation"] if isinstance(entry, dict) else entry
        if not torch.is_tensor(mat):
            raise TypeError(f"layer {li} rotation is {type(mat)!r}, not a tensor")
    else:
        # Legacy flat layouts, kept so an older checkpoint still loads.
        mat = None
        for k in ("rotation", "matrix", "R", "weight"):
            if k in obj and torch.is_tensor(obj[k]):
                mat = obj[k]
                break
        if mat is None:
            for v in obj.values():
                if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] == v.shape[1]:
                    mat = v
                    break
        if mat is None:
            raise ValueError(
                f"no rotation matrix in OSCAR payload with keys {sorted(obj)}"
            )

    if mat.dim() != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"rotation for layer {layer_idx} is {tuple(mat.shape)}, not square")
    if head_dim is not None and mat.shape[0] != head_dim:
        raise ValueError(
            f"rotation for layer {layer_idx} is {tuple(mat.shape)} but head_dim is "
            f"{head_dim}; this payload belongs to a different checkpoint"
        )

    # Say so, once per payload.  An identity rotation is a legal configuration,
    # so its absence can never be detected from a log that only reports errors;
    # this is the line whose absence hid the bug.  Printed from the worker, so
    # it lands in the training log next to the other [mopd]/[qat-train] banners.
    if not any(k[0] == id(obj) for k in _ROT_CACHE):
        n = len(layers) if layers is not None else 1
        orth = (mat.float() @ mat.float().T - torch.eye(mat.shape[0])).abs().max().item()
        print(
            f"[oscar-fake-quant] rotation ACTIVE: objective={obj.get('objective')!r} "
            f"layers={n} head_dim={mat.shape[0]} max|RR^T-I|={orth:.2e}",
            flush=True,
        )

    _ROT_CACHE[ck] = (obj, mat)
    return mat
