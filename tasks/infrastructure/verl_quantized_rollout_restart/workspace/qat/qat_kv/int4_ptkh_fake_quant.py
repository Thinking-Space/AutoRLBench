"""Differentiable fake quantizer for vLLM's ``int4_per_token_head`` KV cache.

Mirrors ``vllm/v1/attention/ops/int4_per_token_head.py`` (vLLM 0.25.1) bit for
bit on the forward path.  The reference is two functions there:

``reshape_and_cache_int4``      RHT in fp32, cast back to the input dtype, then
                                ``_reshape_cache_int4_kernel``: per (token, head)
                                asymmetric INT4 with the zero point steganographed
                                into the scale's low 4 mantissa bits.
``unified_attention_int4``      rotates Q, divides ``softmax_scale`` by
                                ``head_size`` and inverse-rotates the output.

Why the Q side needs no counterpart here
----------------------------------------
The forward transform is the *unnormalized* Walsh-Hadamard ``H`` times a sign
diagonal ``D1``.  Because ``H^T H = d I`` and ``D1^2 = I``:

    <H D1 q, H D1 k_hat_rot> = d * <q, D1 H^T k_hat_rot / d>

so rotating Q, scaling the softmax by ``1/d`` and inverse-rotating the output
collapses exactly to "leave Q alone, and replace K/V by
``invRHT(quant(RHT(x))) / d``".  That is what this module computes.  Training
therefore must not touch Q or the attention scale.

Two things this module deliberately does NOT reuse from vLLM
------------------------------------------------------------
1. ``fast_hadamard_transform``.  Its fast paths call ``_C::hadacore_transform``,
   which has **no registered autograd kernel**.  Backward still runs, the
   gradients are finite, and PyTorch emits only a ``UserWarning`` -- a training
   run would converge on silently wrong gradients.  Its Triton path additionally
   demotes fp32 to bf16.  The butterfly below is the same transform in pure
   torch ops (it is a transcription of that file's own Tier 3 fallback), so it
   is differentiable and exact in fp32.
2. ``qat_kv.naive_kv_fake_quant``.  That one has no rotation, no scale
   truncation, and hooks ``cache.update`` rather than the post-RoPE attention
   interface.  It quantizes a different tensor with a different rule.

Nothing here falls back.  A non-power-of-two head_dim, a mismatched value head
dim, or an unknown preset raises, because the failure mode this whole effort
exists to prevent is a quantizer that silently disagrees with the deployed one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

__all__ = [
    "INT4_PTKH_PRESETS",
    "Int4PTKHConfig",
    "int4_ptkh_fake_quant",
    "int4_ptkh_codes",
    "int4_ptkh_fake_quant_tensor",
    "is_int4_ptkh_scheme",
    "rht",
    "walsh_hadamard",
]

# Only names with a real vLLM inference path belong here; a training-only preset
# could not be parity-tested and a model trained for it could not be deployed.
INT4_PTKH_PRESETS: dict[str, dict] = {
    "int4_per_token_head": {},
}

# vllm .../int4_per_token_head.py::_get_rht_signs
_RHT_SEED_BASE = 0x9E3779B9
_RHT_SEED_STRIDE = 0x517CC1B7

# vllm's asymmetric INT4 grid: 4 bits, [0, 15], scale floored at 1e-6.
_INT4_MAX = 15.0
_SCALE_FLOOR = 1e-6

_SIGN_CACHE: dict[tuple, torch.Tensor] = {}


def is_int4_ptkh_scheme(scheme: Optional[str]) -> bool:
    return bool(scheme) and str(scheme) in INT4_PTKH_PRESETS


@dataclass(frozen=True)
class Int4PTKHConfig:
    """Shape/knob container.  Duck-types the fields the attention hook reads."""

    head_dim: int = 128
    # vLLM allows head_size_v != head_size, but ``unified_attention_int4``
    # divides the inverse-rotated output by ``head_size`` (Q's) while rotating
    # over ``head_size_v`` coordinates.  Those two only agree when the dims are
    # equal, so anything else is refused rather than approximated.
    head_dim_v: Optional[int] = None
    rht_round_idx: int = 0
    # Return K/V in the kernel's rotated basis instead of the original one.
    # Parity testing only -- attention needs the original basis.
    return_rotated: bool = False

    # Read by TurboQuantAttnState.metadata / buffers_for; kept as plain fields
    # so the shared hook needs no isinstance checks for them.
    scheme_name: str = "int4_per_token_head"
    key_quant_bits: int = 4
    value_quant_bits: int = 4
    norm_correction: bool = False

    def __post_init__(self) -> None:
        d = self.head_dim
        if d <= 0 or (d & (d - 1)) != 0:
            raise ValueError(
                f"head_dim must be a positive power of two for the Hadamard "
                f"rotation, got {d}"
            )
        dv = self.head_dim if self.head_dim_v is None else self.head_dim_v
        if dv != d:
            raise ValueError(
                f"head_dim_v ({dv}) != head_dim ({d}); vLLM's "
                "unified_attention_int4 divides the inverse-rotated output by "
                "head_size while rotating over head_size_v, so the two paths "
                "only agree when the dims match"
            )
        if self.rht_round_idx < 0:
            raise ValueError(f"rht_round_idx must be >= 0, got {self.rht_round_idx}")
        if self.scheme_name not in INT4_PTKH_PRESETS:
            raise ValueError(
                f"unknown int4 preset {self.scheme_name!r}; "
                f"valid: {sorted(INT4_PTKH_PRESETS)}"
            )

    @classmethod
    def from_preset(cls, name: str, head_dim: int, **kw) -> "Int4PTKHConfig":
        if name not in INT4_PTKH_PRESETS:
            raise ValueError(
                f"unknown int4 preset {name!r}; valid: {sorted(INT4_PTKH_PRESETS)}"
            )
        return cls(head_dim=head_dim, scheme_name=name, **INT4_PTKH_PRESETS[name], **kw)


# ----------------------------------------------------------------------
# Hadamard / RHT -- differentiable transcription of vLLM's Tier 3 butterfly
# ----------------------------------------------------------------------


def walsh_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized Walsh-Hadamard over the last dim; ``WHT(WHT(x)) == d*x``.

    Transcribed from ``int4_per_token_head.fast_hadamard_transform``'s PyTorch
    fallback.  Pure torch ops, so autograd is exact -- unlike the hadacore and
    Triton tiers that function prefers, neither of which may be used in
    training (see the module docstring).
    """
    d = x.shape[-1]
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"last dim must be a positive power of two, got {d}")
    h = 1
    while h < d:
        xv = x.view(*x.shape[:-1], d // (2 * h), 2, h)
        a = xv[..., 0, :]
        b = xv[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(x.shape)
        h <<= 1
    return x


def _rht_signs(d: int, round_idx: int, device, dtype: torch.dtype) -> torch.Tensor:
    """The +-1 diagonal, generated exactly as vLLM's ``_get_rht_signs`` does.

    Same CPU generator and same seed, so training and inference apply the
    identical rotation.  A drift here would be invisible: both sides would
    still "work", the student would just be trained for a different basis.
    """
    key = (d, round_idx, str(device), dtype)
    signs = _SIGN_CACHE.get(key)
    if signs is None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_RHT_SEED_BASE + round_idx * _RHT_SEED_STRIDE)
        raw = torch.bernoulli(torch.full((d,), 0.5, device="cpu"), generator=gen)
        signs = (2.0 * raw - 1.0).to(device=device, dtype=dtype)
        _SIGN_CACHE[key] = signs
    return signs


def rht(x: torch.Tensor, *, inverse: bool = False, round_idx: int = 0) -> torch.Tensor:
    """``H @ D1 @ x`` (or its transpose), matching vLLM's ``single_rht``.

    Note the asymmetry, which is vLLM's and is reproduced rather than "fixed":
    the inverse applies the signs *after* the transform and does not divide by
    ``d``.  The ``1/d`` lives at the call site.
    """
    d1 = _rht_signs(x.shape[-1], round_idx, x.device, x.dtype)
    if inverse:
        return walsh_hadamard(x) * d1
    return walsh_hadamard(x * d1)


# ----------------------------------------------------------------------
# Quantization
# ----------------------------------------------------------------------


def _ste(quantized: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
    """Straight-through: forward uses ``quantized``, backward is identity."""
    return continuous + (quantized - continuous).detach()


def _round_half_away(x: torch.Tensor) -> torch.Tensor:
    """Round half away from zero, as the kernel's int32 truncation does.

    Triton computes ``(x + 0.5).to(tl.int32)`` for ``x >= 0`` and
    ``(x - 0.5).to(tl.int32)`` otherwise; the int32 cast truncates toward zero,
    so the pair is round-half-away-from-zero.  ``torch.round`` is banker's
    rounding and disagrees on every exact .5, which for a 16-level grid is not
    a rare tie.
    """
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def _truncate_scale_mantissa(scale: torch.Tensor) -> torch.Tensor:
    """Reproduce the scale that the *read* kernel will actually see.

    The store kernel packs the 4-bit zero point into the scale's low 4 mantissa
    bits (``(bits & -16) | zp``) and the read kernel recovers the scale as
    ``bits & -16`` -- clearing the nibble, not restoring it.  So the write/read
    round trip is exactly ``bits & -16``, independent of the zero point, and
    values are quantized with the full scale but dequantized with a truncated
    one.  That asymmetry is part of the real error budget, not an artifact.

    Bit twiddling has no derivative; the truncation is passed through with a
    straight-through estimator so gradients still reach ``x_max``/``x_min``.
    """
    if scale.dtype != torch.float32:
        raise TypeError(f"scale must be float32 to bitcast, got {scale.dtype}")
    bits = scale.detach().contiguous().view(torch.int32)
    recovered = (bits & -16).view(torch.float32)
    return _ste(recovered, scale)


def int4_ptkh_codes(
    x: torch.Tensor, cfg: Int4PTKHConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(codes, zero_point, scale_as_read)`` for one ``[..., head_dim]`` tensor.

    Everything the store kernel writes, in the order and precision vLLM uses:
      1. RHT in fp32, then cast back to ``x.dtype`` -- ``reshape_and_cache_int4``
         does ``single_rht(key.float()).to(key.dtype)``, so on a bf16 model the
         rotated K is rounded to bf16 *before* quantization.
      2. per (token, head) asymmetric INT4 in fp32 (the kernel loads with
         ``.to(tl.float32)``).

    ``codes`` and ``zero_point`` are float tensors on the integer grid [0, 15],
    not ints, so the straight-through gradient survives.  Exposed separately
    from the fake-quant entry point so the GPU parity test can compare against
    the kernel's packed nibbles bit for bit, rather than only comparing
    dequantized values where a compensating pair of errors could hide.
    """
    if x.shape[-1] != cfg.head_dim:
        raise ValueError(
            f"last dim {x.shape[-1]} != cfg.head_dim {cfg.head_dim}; the "
            "quantizer would group the wrong coordinates"
        )
    x_rot = rht(x.float(), round_idx=cfg.rht_round_idx).to(x.dtype).float()

    x_min = x_rot.amin(dim=-1, keepdim=True)
    x_max = x_rot.amax(dim=-1, keepdim=True)
    scale = torch.clamp((x_max - x_min) / _INT4_MAX, min=_SCALE_FLOOR)
    zp_cont = -x_min / scale
    zp = torch.clamp(_ste(_round_half_away(zp_cont), zp_cont), 0.0, _INT4_MAX)

    # ``x * (1/scale)`` because that is the form the kernel writes
    # (``inv_k = 1.0 / k_scale``).  Do not expect it to buy accuracy: divide,
    # reciprocal-multiply and addcmul were measured against the kernel on the
    # same 4M codes and disagreed on 4.11e-3, 4.10e-3 and 4.04e-3 of them --
    # indistinguishable.  The floor is not the arithmetic form.  The rotated
    # K/V arrive in bf16, only 8 mantissa bits, so a few per thousand land
    # *exactly* on a .5 rounding boundary and Triton's fp32 evaluation order
    # decides the tie.  The zero point, computed with a plain divide on both
    # sides, disagrees at the same 4.06e-3 -- which is how we know it is the
    # ties and not the operator.  Every such mismatch is one level.
    q_cont = x_rot * scale.reciprocal() + zp
    q = torch.clamp(_ste(_round_half_away(q_cont), q_cont), 0.0, _INT4_MAX)

    # The read kernel recovers the scale with its low nibble cleared, so that
    # -- not the scale used to quantize -- is what dequantization must use.
    return q, zp, _truncate_scale_mantissa(scale)


def int4_ptkh_fake_quant_tensor(
    x: torch.Tensor, cfg: Int4PTKHConfig
) -> torch.Tensor:
    """Fake-quantize one ``[..., head_dim]`` tensor through the INT4 path.

    Quantize (see ``int4_ptkh_codes``), dequantize with the truncated scale,
    then inverse-rotate and divide by ``head_dim``.
    """
    orig_dtype = x.dtype
    q, zp, scale_read = int4_ptkh_codes(x, cfg)
    x_hat_rot = (q - zp) * scale_read

    if cfg.return_rotated:
        return x_hat_rot.to(orig_dtype)

    # 4. undo the rotation; the 1/d is the norm of the unnormalized transform,
    #    applied at inference by softmax_scale/head_size and the output divide.
    x_hat = rht(x_hat_rot, inverse=True, round_idx=cfg.rht_round_idx) / cfg.head_dim
    return x_hat.to(orig_dtype)


def int4_ptkh_fake_quant(
    key: torch.Tensor, value: torch.Tensor, cfg: Int4PTKHConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake-quantize a (key, value) pair.  Both are ``[..., head_dim]``.

    K and V take the identical path: ``_reshape_cache_int4_kernel`` runs the
    same min/max/zp/pack block twice, once per stream, and both are rotated by
    ``reshape_and_cache_int4``.
    """
    return (
        int4_ptkh_fake_quant_tensor(key, cfg),
        int4_ptkh_fake_quant_tensor(value, cfg),
    )
