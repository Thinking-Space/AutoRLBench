"""Fix the OSCAR BF16 window's V for models whose k and v have different strides.

The bug
-------
``_oscar_staging_write_kernel`` (oscar_attn.py) loads K and V with ONE set of
strides, K's::

    k = tl.load(key_ptr + t * stride_kt + h * stride_kh + d, ...)
    v = tl.load(val_ptr + t * stride_kt + h * stride_kh + d, ...)   # <- K's strides

That is correct whenever k and v are laid out identically, which they are on
Qwen2: both are column slices of the fused ``qkv`` projection, row stride
``(Hq+2Hk)*D``.  On Qwen3 the k that reaches attention has been through the
per-head ``k_norm`` -- a fresh contiguous ``[N, Hk*D]`` tensor, row stride
``Hk*D`` -- while v is still the ``qkv`` slice.  The kernel then walks V with
a row stride 6x too small on Qwen3-8B (1024 instead of 6144) and stages the
wrong bytes as every token's value.

What it does to the model
-------------------------
The attention *pattern* is right (K is right), the *values* are garbage, for
every token inside the BF16 sink/recent window, on every quantized layer.
Measured on Qwen3-8B with nothing quantized at all (recent window = context,
no rotation): decode attention output NMSE 1-10 against dense attention over
the raw K/V (jobs 59268/59279); the 1.5B (Qwen2) on the same code: 3e-6.
Text output is garbage from the third generated token (jobs 59163-59166), and
every Qwen3-4B/8B OSCAR number recorded before 2026-09-16 was measured through
this bug.  The INT2 copy is not affected: ``do_kv_cache_update`` rotates V
into a fresh tensor before ``oscar_store``.

The fix
-------
The per-request staging kernel (``oscar_staging_patch.py``) now takes V's own
strides.  For the stock shared-arena path the kernel lives in the shared
install, which is never edited, so this module wraps ``_staging_write`` and
hands the kernel a V whose strides equal K's -- a ``.contiguous()`` copy of
``[N, Hk, D]`` when they differ (K is contiguous on every model that hits
this; if it ever is not, both are copied).  Cost: one small copy per layer per
step, only on affected models.

Always on -- there is no correct reason to stage V through K's strides.
"""

from __future__ import annotations

import os

__all__ = ["apply"]

_DONE = False


def apply(verbose: bool = True) -> bool:
    global _DONE
    if _DONE:
        return True
    if os.environ.get("QAT_OSCAR_V_STRIDE_FIX", "1").strip() == "0":
        return False

    from vllm.v1.attention.backends import oscar_attn as OA

    Impl = OA.OscarAttentionImpl
    orig = Impl._staging_write
    seen = {"reported": False}

    def _staging_write(self, layer, key, value, attn_metadata):
        if key.stride() != value.stride():
            if verbose and not seen["reported"]:
                seen["reported"] = True
                print(f"[oscar-vstride] pid={os.getpid()} k/v strides differ "
                      f"(k={tuple(key.stride())}, v={tuple(value.stride())}); "
                      f"staging V through a contiguous copy", flush=True)
            if not key.is_contiguous():
                key = key.contiguous()
            value = value.contiguous()
        return orig(self, layer, key, value, attn_metadata)

    Impl._staging_write = _staging_write
    _DONE = True
    return True
