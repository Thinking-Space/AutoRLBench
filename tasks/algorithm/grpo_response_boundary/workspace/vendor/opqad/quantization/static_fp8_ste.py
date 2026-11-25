"""Static FP8 fake quantization with an identity-gradient STE."""
from __future__ import annotations

import math
import torch


def static_fp8_quant_dequant_ste(
    x: torch.Tensor, *, scale: float = 1.0, dtype: torch.dtype = torch.float8_e4m3fn
) -> torch.Tensor:
    """Match vLLM QuantFP8 static quantization and dequantize to ``x.dtype``.

    vLLM uses ``x * reciprocal(scale)``, clamps to the FP8 finite range, then
    casts to E4M3. This function dequantizes that value by multiplying by the
    same scale and uses an identity straight-through gradient.
    """
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("static FP8 scale must be finite and positive")
    if dtype != torch.float8_e4m3fn:
        raise ValueError("only vLLM FP8 E4M3 static quantization is supported")
    original_dtype = x.dtype
    scale_t = torch.tensor(scale, dtype=torch.float32, device=x.device)
    quantized = (x.to(torch.float32) * torch.reciprocal(scale_t)).clamp(-448.0, 448.0).to(dtype)
    dequantized = (quantized.to(torch.float32) * scale_t).to(original_dtype)
    return dequantized.detach() + (x - x.detach())
