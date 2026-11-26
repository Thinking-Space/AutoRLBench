"""Differentiable (STE) reference implementation of vLLM's TurboQuant KV quantizer.

This mirrors, operation for operation, the deployed inference path:

  store   vllm/v1/attention/ops/triton_turboquant_store.py
  dequant vllm/v1/attention/ops/triton_turboquant_decode.py::_tq_full_dequant_kv
  buffers vllm/v1/attention/backends/turboquant_attn.py::_ensure_on_device

The point of this module is QAT: putting the *deployed* quantizer into the
training graph so gradients see it.  Fidelity is therefore the whole
requirement, and it is checked by ``tests/test_turboquant_parity.py``, which
runs the real Triton kernels and compares element-wise.  Do not "improve" the
arithmetic here -- any deviation silently trains against a quantizer that is
not the one being deployed.

Algorithm (per (token, kv_head), on head_dim-sized vectors)
-----------------------------------------------------------
Keys, MSE path::

    norms  = ||k||                         # stored as fp16
    x_hat  = k / (norms + 1e-8)            # unit sphere
    y      = x_hat @ H                     # orthonormal Sylvester Hadamard
    idx    = searchsorted(midpoints, y)    # Lloyd-Max, 2^bits levels
    c      = centroids[idx]
    c      = c / sqrt(sum(c^2) + 1e-16)    # iff norm_correction
    k_rot  = fp16(norms) * c               # <- what the kernel reconstructs

Note the kernel stops there: it never applies the inverse rotation, because
attention rotates Q instead (H is orthonormal so q.k is preserved).  A training
forward needs K back in the original basis, so we finish with ``k_rot @ H``
(H is symmetric and orthonormal, hence its own inverse).  ``return_rotated``
exposes the kernel-space tensor for parity testing.

Values, uniform path::

    lo, hi = min(v), max(v)
    scale  = max((hi - lo) / (2^bits - 1), 1e-8)      # fp32 for quantizing
    q      = clamp(floor((v - lo)/scale + 0.5), 0, 2^bits - 1)
    v_hat  = q * fp16(scale) + fp16(lo)               # fp16 for dequantizing

The scale/zero asymmetry is deliberate and matches the kernel: quantization
uses the fp32 values, reconstruction uses the fp16 round-tripped ones.

Gradients
---------
Only two operations are non-differentiable: the key ``searchsorted`` and the
value ``floor``.  Both get a straight-through estimator; everything else
(normalisation, rotation, norm correction, min/max, scale) carries a real
gradient.  This matters at 3 bits, where a whole-quantizer STE would be a much
cruder approximation than is necessary.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

__all__ = [
    "TQ_FAKE_QUANT_PRESETS",
    "TurboQuantFakeQuantConfig",
    "build_hadamard",
    "lloyd_max_centroids",
    "turboquant_fake_quant",
    "turboquant_fake_quant_keys",
    "turboquant_fake_quant_values",
    "turboquant_key_indices",
    "turboquant_key_rotate",
    "turboquant_value_levels",
    "turboquant_value_scale_zero",
]

# Mirrors vllm .../quantization/turboquant/config.py::TQ_PRESETS exactly.
# Only presets with a real vLLM inference path belong here: a training-only
# preset could not be parity-tested, and a model trained for it could not be
# deployed.
TQ_FAKE_QUANT_PRESETS: dict[str, dict] = {
    "turboquant_4bit_nc": {"key_quant_bits": 4, "value_quant_bits": 4, "norm_correction": True},
    "turboquant_k3v4_nc": {"key_quant_bits": 3, "value_quant_bits": 4, "norm_correction": True},
    "turboquant_3bit_nc": {"key_quant_bits": 3, "value_quant_bits": 3, "norm_correction": True},
    # The 2-bit presets were held back until the Triton store and decode
    # kernels carried a 2-bit branch, because this table's contract is that
    # every entry is deployable.  Both kernels now have that branch and the
    # vLLM inference path produced real AIME24 avg@32 numbers for these two
    # presets (k2v3 15.21, 2bit 0.73 against bf16 57.60), so they satisfy the
    # contract and are admitted.
    "turboquant_k3v2_nc": {"key_quant_bits": 3, "value_quant_bits": 2, "norm_correction": True},
    "turboquant_k2v3_nc": {"key_quant_bits": 2, "value_quant_bits": 3, "norm_correction": True},
    "turboquant_2bit_nc": {"key_quant_bits": 2, "value_quant_bits": 2, "norm_correction": True},
}

# vLLM hard-codes this in TurboQuantConfig.get_boundary_skip_layers; without it
# GSM8K reportedly drops ~30 points on the aggressive presets.  Training must
# reproduce the same layer set or it fake-quantizes layers that run in fp at
# inference time.
BOUNDARY_SKIP_LAYERS = 2


@dataclass(frozen=True)
class TurboQuantFakeQuantConfig:
    head_dim: int = 128
    key_quant_bits: int = 3
    value_quant_bits: int = 3
    norm_correction: bool = True
    # Return keys in the kernel's rotated basis instead of the original one.
    # Only used by the parity test.
    return_rotated: bool = False

    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if self.key_quant_bits not in (2, 3, 4):
            # FP8 keys (key_quant_bits=8, the k8v4 preset) take a completely
            # different store path with no rotation or centroids; it is not
            # implemented rather than silently approximated.
            raise ValueError(
                f"key_quant_bits must be 2, 3 or 4, got {self.key_quant_bits} "
                "(FP8 keys are not implemented)"
            )
        if self.value_quant_bits not in (2, 3, 4):
            raise ValueError(f"value_quant_bits must be 2, 3 or 4, got {self.value_quant_bits}")

    @property
    def n_key_centroids(self) -> int:
        return 2**self.key_quant_bits

    @property
    def n_value_levels(self) -> int:
        return 2**self.value_quant_bits

    @classmethod
    def from_preset(cls, name: str, head_dim: int, **overrides) -> "TurboQuantFakeQuantConfig":
        if name not in TQ_FAKE_QUANT_PRESETS:
            valid = ", ".join(sorted(TQ_FAKE_QUANT_PRESETS))
            raise ValueError(f"unknown TurboQuant preset {name!r}; valid: {valid}")
        preset = dict(TQ_FAKE_QUANT_PRESETS[name])
        preset.update(overrides)
        return cls(head_dim=head_dim, **preset)


def skip_layer_indices(num_layers: int, n: int = BOUNDARY_SKIP_LAYERS) -> list[int]:
    """Layer indices vLLM leaves in full precision (first n and last n)."""
    if n <= 0 or num_layers <= 0:
        return []
    n = min(n, num_layers // 2)
    return sorted(set(range(n)) | set(range(num_layers - n, num_layers)))


def oscar_quant_all_layers() -> bool:
    """OSCAR quantizes every layer. Always. This is not a knob.

    The boundary skip {0, 1, L-2, L-1} belongs to the shared vLLM port, not to
    OSCAR: the paper quantizes all layers and its own ablation says boundary
    protection is unnecessary. For a while this repo exposed the choice as
    ``VLLM_OSCAR_QUANT_ALL_LAYERS`` and it cost, in order:

      * every landed OSCAR bit rate (quoted 2.14, actually 4.33 on the 1.5B and
        3.90 on the 4B, because four layers ran BF16 and nobody set the flag);
      * job 63401, dead at init, when the default reached the rollout but not
        the training hook;
      * jobs 63686/63687/63817, three full training arms at skip-4 -- roughly
        double the bit budget of the hybrid (2.76) and KIVI (3.01) baselines
        they exist to be compared against, so their numbers cannot go in the
        table at all.

    Each of those is the same failure: a default silently decided which
    experiment ran. So the choice is gone. The variable is still *read*, and
    only to refuse -- a launcher that still passes 0 stops loudly instead of
    quietly running the other experiment.

    ``opqad/evaluation/kv_bits.py`` keeps its ``skip_layers`` argument: it describes
    archived runs, and reporting what a past run did is not the same as
    offering to do it again.
    """
    raw = os.environ.get("VLLM_OSCAR_QUANT_ALL_LAYERS", "").strip()
    if raw not in ("", "1"):
        raise ValueError(
            f"VLLM_OSCAR_QUANT_ALL_LAYERS={raw!r} is no longer honoured: OSCAR "
            "quantizes every layer, unconditionally. The boundary skip is a "
            "property of the upstream vLLM port, not an experimental variable, "
            "and three training arms were lost to it being one. Drop the "
            "variable. To describe an archived skip-4 run, pass skip_layers=4 "
            "to opqad/evaluation/kv_bits.py instead."
        )
    return True



@functools.lru_cache(maxsize=16)
def _hadamard_cpu(d: int) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard, identical to vLLM's _build_hadamard_cached."""
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"head_dim must be a power of two for the Hadamard rotation, got {d}")
    h = torch.tensor([[1.0]], dtype=torch.float64)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(d)).to(torch.float32)


def build_hadamard(d: int, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return _hadamard_cpu(d).to(device=device, dtype=dtype)


@functools.lru_cache(maxsize=16)
def _centroids_cpu(d: int, bits: int) -> torch.Tensor:
    """Lloyd-Max centroids for N(0, 1/d).

    Imported from vLLM when available so training and inference cannot drift
    apart.  The fallback is a line-for-line copy of the same solver, used only
    when vLLM is not importable (e.g. a CPU-only unit-test environment); the
    parity test asserts the two agree.
    """
    try:
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            get_centroids,
        )

        return get_centroids(d, bits).detach().to(torch.float32).cpu()
    except Exception:
        return _solve_lloyd_max_fallback(d, bits)


def _solve_lloyd_max_fallback(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10):
    n_levels = 2**bits
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)

    def pdf(x: float) -> float:
        return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))

    def trapz(f, a: float, b: float, n: int = 200) -> float:
        h = (b - a) / n
        acc = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            acc += f(a + i * h)
        return acc * h

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            den = trapz(pdf, a, b)
            new.append(trapz(lambda x: x * pdf(x), a, b) / den if den > 1e-15 else centroids[i])
        if max(abs(new[i] - centroids[i]) for i in range(n_levels)) < tol:
            centroids = new
            break
        centroids = new
    return torch.tensor(centroids, dtype=torch.float32)


def lloyd_max_centroids(d: int, bits: int, device=None) -> torch.Tensor:
    return _centroids_cpu(d, bits).to(device=device)


def _midpoints(centroids: torch.Tensor) -> torch.Tensor:
    c, _ = centroids.sort()
    return (c[:-1] + c[1:]) / 2


def _ste(quantized: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
    """Straight-through: forward uses ``quantized``, backward is identity."""
    return continuous + (quantized - continuous).detach()


def _fp16_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """fp16 store/load, straight-through in backward."""
    return _ste(x.to(torch.float16).to(x.dtype), x)


def turboquant_key_rotate(
    key: torch.Tensor,
    cfg: "TurboQuantFakeQuantConfig",
    *,
    hadamard: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The store kernel's pre-GEMM: returns ``(y, norms)``.

    Mirrors triton_turboquant_store.triton_turboquant_store's MSE path::

        norms = k.norm(dim=1, keepdim=True); x_hat = k/(norms+1e-8); y = x_hat @ PiT
    """
    d = cfg.head_dim
    k = key.float()
    h = build_hadamard(d, k.device) if hadamard is None else hadamard.to(k.device, torch.float32)
    norms = torch.linalg.vector_norm(k, dim=-1, keepdim=True)
    x_hat = k / (norms + 1e-8)
    return x_hat @ h, norms


def turboquant_key_indices(
    y: torch.Tensor,
    cfg: "TurboQuantFakeQuantConfig",
    *,
    centroids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lloyd-Max centroid indices for rotated keys ``y``.

    The kernel's binary search is ``lo = where(y >= mid_val, mid+1, lo)``, i.e.
    the count of midpoints <= y, which is ``searchsorted(..., right=True)``.
    """
    d = cfg.head_dim
    c_all = (
        lloyd_max_centroids(d, cfg.key_quant_bits, y.device)
        if centroids is None
        else centroids.to(y.device, torch.float32)
    )
    mids = _midpoints(c_all)
    flat = y.reshape(-1, d).contiguous().float()
    idx = torch.searchsorted(mids.contiguous(), flat, right=True)
    return idx.clamp_(max=cfg.n_key_centroids - 1).reshape(y.shape)


def turboquant_value_scale_zero(
    value: torch.Tensor,
    cfg: "TurboQuantFakeQuantConfig",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(zero, scale)`` in fp32, exactly as the store kernel computes them."""
    v = value.float()
    levels = cfg.n_value_levels - 1
    v_min = v.min(dim=-1, keepdim=True).values
    v_max = v.max(dim=-1, keepdim=True).values
    scale = (v_max - v_min) / levels
    # kernel: tl.where(v_scale > 1e-8, v_scale, 1e-8) -- a select, not a max,
    # but identical for the non-negative scales produced here.
    scale = torch.where(scale > 1e-8, scale, torch.full_like(scale, 1e-8))
    return v_min, scale


def turboquant_value_levels(
    value: torch.Tensor,
    cfg: "TurboQuantFakeQuantConfig",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(q, zero, scale)``; ``q`` are the integer levels the kernel packs."""
    v = value.float()
    levels = cfg.n_value_levels - 1
    v_min, scale = turboquant_value_scale_zero(v, cfg)
    q = torch.floor((v - v_min) / scale + 0.5).clamp(0, levels).to(torch.int64)
    return q, v_min, scale


def turboquant_fake_quant_keys(
    key: torch.Tensor,
    cfg: TurboQuantFakeQuantConfig,
    *,
    hadamard: Optional[torch.Tensor] = None,
    centroids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fake-quantize keys. ``key`` is ``[..., head_dim]``; returns the same shape."""
    d = cfg.head_dim
    if key.shape[-1] != d:
        raise ValueError(f"key last dim {key.shape[-1]} != head_dim {d}")

    orig_dtype = key.dtype
    k = key.float()

    h = build_hadamard(d, k.device) if hadamard is None else hadamard.to(k.device, torch.float32)
    c_all = (
        lloyd_max_centroids(d, cfg.key_quant_bits, k.device)
        if centroids is None
        else centroids.to(k.device, torch.float32)
    )
    mids = _midpoints(c_all)

    y, norms = turboquant_key_rotate(k, cfg, hadamard=h)
    idx = turboquant_key_indices(y, cfg, centroids=c_all)
    # The straight-through estimator belongs on the *index selection*, not on
    # the table being indexed.  ``_ste(c_all[idx], y)`` puts the whole gather
    # inside ``.detach()``, which makes d c / d c_all exactly zero -- so a
    # learnable centroid table could never move.  This form is identical in
    # the forward pass and identical in d/dy, but lets the table receive its
    # true (exact, non-STE) gather gradient.
    c = c_all[idx] + (y - y.detach())

    if cfg.norm_correction:
        c = c / torch.sqrt((c * c).sum(dim=-1, keepdim=True) + 1e-16)

    k_rot = _fp16_roundtrip(norms) * c
    out = k_rot if cfg.return_rotated else k_rot @ h
    return out.to(orig_dtype)


def turboquant_fake_quant_values(
    value: torch.Tensor,
    cfg: TurboQuantFakeQuantConfig,
) -> torch.Tensor:
    """Fake-quantize values. ``value`` is ``[..., head_dim]``; same shape out."""
    d = cfg.head_dim
    if value.shape[-1] != d:
        raise ValueError(f"value last dim {value.shape[-1]} != head_dim {d}")

    orig_dtype = value.dtype
    v = value.float()
    levels = cfg.n_value_levels - 1

    v_min, scale = turboquant_value_scale_zero(v, cfg)
    x = (v - v_min) / scale
    # ((x) + 0.5).to(int32) truncates toward zero; x >= 0 here, so it is
    # round-half-up -- NOT torch.round, which rounds halves to even.
    q = _ste(torch.floor(x + 0.5), x).clamp(0, levels)

    # Quantize with fp32 scale/min, reconstruct with the fp16 stored copies.
    return (q * _fp16_roundtrip(scale) + _fp16_roundtrip(v_min)).to(orig_dtype)


def turboquant_fake_quant(
    key: torch.Tensor,
    value: torch.Tensor,
    cfg: TurboQuantFakeQuantConfig,
    *,
    hadamard: Optional[torch.Tensor] = None,
    centroids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake-quantize a (key, value) pair. Both are ``[..., head_dim]``."""
    return (
        turboquant_fake_quant_keys(key, cfg, hadamard=hadamard, centroids=centroids),
        turboquant_fake_quant_values(value, cfg),
    )
