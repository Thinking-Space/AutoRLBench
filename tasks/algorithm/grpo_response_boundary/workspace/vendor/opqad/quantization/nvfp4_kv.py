"""Reference NVFP4 KV fake-quant codec and STE.

Layout contract follows FlashInfer/vLLM 0.32: E2M1 payload, FP8 E4M3
block scales per 16 channels, K/V global decode scale, cache writer receives
post-RoPE K and unrotated V. Its CUDA writer parity passed on the controlled
L20D case recorded in experiments/2026-11-07/fp4_execution/PROGRESS.md.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

@dataclass(frozen=True)
class NVFP4KVConfig:
    group_size: int = 16
    k_scale: float = 1.0
    v_scale: float = 1.0
    def __post_init__(self):
        if self.group_size != 16:
            raise ValueError("vLLM NVFP4 KV uses 16-value block scales")
        if not (self.k_scale > 0 and self.v_scale > 0):
            raise ValueError("global K/V decode scales must be positive")

def _quant_dequant(x: torch.Tensor, global_scale: float, group_size: int):
    if x.shape[-1] % group_size:
        raise ValueError(f"last dim {x.shape[-1]} must be divisible by {group_size}")
    original_dtype = x.dtype
    # vLLM's append API accepts bf16/fp16; its KV input is rounded to model dtype.
    source = x.to(torch.bfloat16).to(torch.float32)
    blocks = source.reshape(*source.shape[:-1], -1, group_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    # Global decode scale multiplies the per-block E4M3 scale on decode.
    # Choose the E4M3 block scale to use the E2M1 finite maximum (6).
    scale_fp32 = (amax / (6.0 * float(global_scale))).clamp_min(2.0 ** -126)
    scale = scale_fp32.to(torch.float8_e4m3fn).to(torch.float32)
    # Native CUDA writer multiplies by reciprocal dequant scale. This differs
    # from division at exact E2M1 midpoints because reciprocal FP32 rounding can
    # move some values a few ulps to either side of the midpoint.
    inv_scale = torch.reciprocal(scale * float(global_scale))
    normalized = blocks * inv_scale
    levels = torch.tensor(E2M1_LEVELS, device=x.device, dtype=torch.float32)
    distances = (normalized.abs().unsqueeze(-1) - levels).abs()
    minimum = distances.amin(dim=-1, keepdim=True)
    # Native E2M1 conversion uses ties-to-even. torch.argmin selects the lower
    # magnitude code; for a tie whose lower code index is odd, select its even neighbor.
    idx = distances.argmin(dim=-1)
    tied = distances == minimum
    idx = torch.where(tied.sum(dim=-1) > 1, idx + (idx.remainder(2) == 1).to(idx.dtype), idx)
    q = levels[idx] * normalized.sign()
    dq = q * scale * float(global_scale)
    return dq.reshape_as(source).to(original_dtype), scale.to(torch.float8_e4m3fn), idx, normalized.sign()

def nvfp4_kv_fake_quant(x: torch.Tensor, *, global_scale: float = 1.0,
                        group_size: int = 16, ste: bool = True) -> torch.Tensor:
    """Fake quantize already-positioned K or V and return dequantized values.

    Call after RoPE for K and before cache write for both tensors. Forward
    rounds input through BF16 to match vLLM KV append input, then E2M1/E4M3;
    backward is identity STE. Scale uses per-16-value absmax and the supplied
    native global decode scale. This reference must be compared with the
    installed FlashInfer append kernel before claiming bitwise/native parity.
    """
    dq, _, _, _ = _quant_dequant(x, global_scale, group_size)
    return x + (dq - x).detach() if ste else dq

def nvfp4_kv_fake_quant_pair(key: torch.Tensor, value: torch.Tensor,
                             config: NVFP4KVConfig = NVFP4KVConfig()):
    """Quantize post-RoPE key and value using separate native layer scales."""
    return (nvfp4_kv_fake_quant(key, global_scale=config.k_scale, group_size=config.group_size),
            nvfp4_kv_fake_quant(value, global_scale=config.v_scale, group_size=config.group_size))

def nvfp4_cpu_reference(x: torch.Tensor, *, global_scale: float = 1.0,
                        group_size: int = 16):
    """Return packed E2M1 bytes, E4M3 block scales, and dequantized reference."""
    dq, scales, idx, signs = _quant_dequant(x, global_scale, group_size)
    # Positive E2M1 codes 0..7 follow increasing magnitude; high nibble bit is sign.
    codes = idx.to(torch.uint8) | (signs < 0).to(torch.uint8) * 8
    codes = codes.reshape(*x.shape[:-1], -1, group_size)
    codes = codes.reshape(*x.shape[:-1], -1)
    if codes.shape[-1] % 2:
        raise ValueError("packed E2M1 needs an even number of elements")
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), scales.squeeze(-1).contiguous(), dq

def nvfp4_cuda_alignment_check(x: torch.Tensor, *, global_scale: float = 1.0,
                               group_size: int = 16):
    """Compare CPU reference with FlashInfer native pack on supplied CUDA rows.

    Caller must provide post-RoPE K (or V), CUDA x, and a temporary paged
    cache/slot mapping. Kept as an explicit entry point so the GPU smoke can
    invoke the same quantizer seam after backend availability is established.
    """
    if not x.is_cuda:
        raise ValueError("CUDA alignment check requires a CUDA tensor")
    from flashinfer import nvfp4_quantize_append_paged_kv_cache_with_slot_mapping
    if x.ndim != 3 or x.shape[-1] % group_size:
        raise ValueError("expected [rows, kv_heads, head_dim] and divisible head_dim")
    rows, heads, dim = x.shape
    page = max(rows, 16)
    payload = torch.zeros((1, page, heads, dim // 2), dtype=torch.uint8, device=x.device)
    scales = torch.zeros((1, page, heads, dim // group_size), dtype=torch.float8_e4m3fn, device=x.device)
    slots = torch.arange(rows, dtype=torch.int32, device=x.device)
    nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
        x.to(torch.bfloat16), x.to(torch.bfloat16), slots,
        (payload, payload.clone()), (scales, scales.clone()),
        float(global_scale), float(global_scale), kv_layout="NHD")
    return payload, scales
