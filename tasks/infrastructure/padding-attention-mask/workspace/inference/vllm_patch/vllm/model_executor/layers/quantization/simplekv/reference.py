# SPDX-License-Identifier: Apache-2.0
"""Reference pack/unpack for the simple KV slot layout.

This is the byte-level contract the Triton kernels must reproduce.  It exists
separately from the kernels for the same reason TurboQuant keeps a reference
next to its kernels: a quantiser can be self-consistent and still disagree with
the tensor the model actually attends to, and only a bit-for-bit comparison
against an independent implementation catches that.  In this project the 2-bit
TurboQuant presets were absent from the parity list for exactly that reason and
the one layout that differs from every other width went unverified.

Per-token slot layout, per head:

    [ key_data | key_meta | value_data | value_meta ]

``*_data``  ceil(head_dim * bits / 8) bytes of packed codes, low bits first
``*_meta``  4 bytes per channel group: fp16 scale, then fp16 zero point

Per-channel keys do not fit this layout -- their scale is shared across a token
group and therefore belongs to the block, not the token -- and are handled by a
separate path.
"""

from __future__ import annotations

import torch

from .config import SimpleKVConfig


def _pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack ``[..., D]`` integer codes into ``[..., ceil(D*bits/8)]`` uint8."""
    per_byte = 8 // bits
    if 8 % bits:
        raise ValueError(f"bit width {bits} does not divide a byte")
    *lead, d = codes.shape
    grouped = codes.reshape(*lead, d // per_byte, per_byte).to(torch.int32)
    shifts = (torch.arange(per_byte, device=codes.device) * bits).view(
        *([1] * (len(lead) + 1)), per_byte
    )
    return ((grouped << shifts).sum(-1) & 0xFF).to(torch.uint8)


def _unpack_codes(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    shifts = (torch.arange(per_byte, device=packed.device) * bits).view(
        *([1] * packed.ndim), per_byte
    )
    out = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & mask
    return out.reshape(*packed.shape[:-1], d)


def quantize_per_token(x: torch.Tensor, cfg: SimpleKVConfig):
    """Return (codes, scale, zero) for ``[..., D]`` using asymmetric min/max."""
    g = cfg.value_group
    *lead, d = x.shape
    grouped = x.reshape(*lead, d // g, g).to(torch.float32)
    mn = grouped.amin(-1, keepdim=True)
    mx = grouped.amax(-1, keepdim=True)
    max_int = 2**cfg.bits - 1
    scale = (mx - mn) / max_int
    # A constant group has no representable range.  Storing a zero scale would
    # make dequantisation produce the zero point for every element and destroy
    # the value; keep the group exactly instead.
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = ((grouped - mn) / safe).round().clamp(0, max_int)
    return (codes.reshape(*lead, d).to(torch.int32),
            scale.squeeze(-1).to(torch.float16),
            mn.squeeze(-1).to(torch.float16))


def dequantize_per_token(codes, scale, zero, cfg: SimpleKVConfig, dtype):
    g = cfg.value_group
    *lead, d = codes.shape
    grouped = codes.reshape(*lead, d // g, g).to(torch.float32)
    s = scale.to(torch.float32).unsqueeze(-1)
    z = zero.to(torch.float32).unsqueeze(-1)
    # Mirror the encoder: a zero scale meant "kept exactly", and the encoder
    # emitted code 0 for every element, so the zero point is the value.
    s = torch.where(s == 0, torch.zeros_like(s), s)
    return (grouped * s + z).reshape(*lead, d).to(dtype)


def pack_slot(key: torch.Tensor, value: torch.Tensor, cfg: SimpleKVConfig) -> torch.Tensor:
    """Pack one token's key and value (``[..., D]``) into the byte slot."""
    if cfg.key_rule != "per_token":
        raise ValueError("pack_slot covers the per-token key rule only")
    parts = []
    for x in (key, value):
        codes, scale, zero = quantize_per_token(x, cfg)
        meta = torch.stack(
            (scale.view(torch.uint8).reshape(*scale.shape, 2)[..., 0],
             scale.view(torch.uint8).reshape(*scale.shape, 2)[..., 1],
             zero.view(torch.uint8).reshape(*zero.shape, 2)[..., 0],
             zero.view(torch.uint8).reshape(*zero.shape, 2)[..., 1]),
            dim=-1,
        ).reshape(*scale.shape[:-1], -1)
        parts += [_pack_codes(codes, cfg.bits), meta]
    slot = torch.cat(parts, dim=-1)
    pad = cfg.slot_size_aligned - slot.shape[-1]
    if pad:
        slot = torch.cat([slot, torch.zeros(*slot.shape[:-1], pad, dtype=torch.uint8,
                                            device=slot.device)], dim=-1)
    return slot


def unpack_slot(slot: torch.Tensor, cfg: SimpleKVConfig, dtype=torch.bfloat16):
    """Inverse of :func:`pack_slot`; returns dequantised (key, value)."""
    d, ng = cfg.head_dim, cfg.head_dim // cfg.value_group
    kd, km = cfg.key_data_bytes, cfg.key_meta_bytes_per_token
    vd = cfg.value_data_bytes
    out = []
    off = 0
    for data_bytes in (kd, vd):
        codes = _unpack_codes(slot[..., off:off + data_bytes], cfg.bits, d)
        off += data_bytes
        meta = slot[..., off:off + 4 * ng].reshape(*slot.shape[:-1], ng, 4)
        off += 4 * ng
        scale = meta[..., 0:2].reshape(*meta.shape[:-2], ng * 2).contiguous().view(torch.float16)
        zero = meta[..., 2:4].reshape(*meta.shape[:-2], ng * 2).contiguous().view(torch.float16)
        out.append(dequantize_per_token(codes, scale, zero, cfg, dtype))
    return out[0], out[1]
