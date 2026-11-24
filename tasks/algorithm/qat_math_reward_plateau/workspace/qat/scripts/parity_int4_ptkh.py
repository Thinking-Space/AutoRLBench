#!/usr/bin/env python
"""GPU parity: our INT4 fake quantizer vs vLLM's own store kernel.

The unit tests pin the operator's internal consistency; this pins it against
the thing that actually runs at inference.  It writes a real K/V batch through
``reshape_and_cache_int4`` (the Triton store kernel), unpacks the nibbles and
the steganographed zero points back out of the paged cache, and compares.

Two stages, because the two halves of the operator have very different error
budgets and lumping them together hides both:

**strict** -- rotation bypassed on both sides, so the comparison is purely the
    quantizer: min/max, scale, zero point, round-half-away, mantissa
    truncation.  ``scale`` must match *bit for bit*.  Codes get a small
    tolerance: the rotated K/V arrive in bf16, so a few per thousand land
    exactly on a .5 rounding boundary and evaluation order decides the tie.
    All mismatches must be a single level.

**full** -- the whole path, including rotation.  This one cannot be tight, and
    the reason is deliberate rather than a defect.  vLLM rotates with hadacore,
    which computes the Hadamard in bf16 (measured: 2.9e-3 relative against an
    exact fp32 transform).  We must not use that kernel -- it has no registered
    autograd kernel, so backward silently returns wrong gradients -- and we
    would not want to even if we could: its bf16 error is numerical noise of
    the particular kernel build, not a property of the rotation the model is
    served with.  The training side therefore uses an exact fp32 butterfly,
    exactly as ``turboquant_fake_quant`` uses an exact fp32 Hadamard matmul
    against the same inference kernels.  This stage exists to *measure* that
    gap, not to drive it to zero.

Exits non-zero if either stage's bounds are exceeded.
"""

from __future__ import annotations

import argparse
import sys

import torch

# Strict stage: codes may differ only where a bf16 value sits exactly on a .5
# rounding boundary, where Triton's fp32 evaluation order breaks the tie.
# Measured floor on 4M codes: 4.1e-3, and identical for divide,
# reciprocal-multiply and addcmul -- the arithmetic form does not move it.  The
# zero point, a plain divide on both sides, disagrees at the same 4.06e-3,
# which is the evidence that this is ties rather than a wrong operator.  The
# bound is set just above the measured floor; every mismatch must be one level,
# and the scale must still be bit-exact.
STRICT_CODE_TOL = 1e-2
STRICT_ZP_TOL = 1e-2
# Full stage: dominated by the bf16-vs-fp32 rotation described above.
FULL_DEQUANT_REL_TOL = 5e-2


def _unpack(cache, scale_cache, T, H, D):
    """Recover (codes, zp, scale) from the paged cache, as the read kernel does."""
    packed = cache.reshape(T, H, D // 2)
    lo = (packed & 0xF).to(torch.float32)
    hi = ((packed >> 4) & 0xF).to(torch.float32)
    codes = torch.stack([lo, hi], dim=-1).reshape(T, H, D)
    bits = scale_cache.reshape(T, H).view(torch.int32)
    # The low nibble is the zero point; the scale is recovered with that nibble
    # cleared, not restored.
    return codes, (bits & 0xF).to(torch.float32), (bits & -16).view(torch.float32)


def _store(key, value, T, H, D, block):
    """Run the real store kernel and return the unpacked K and V."""
    from vllm.v1.attention.ops.int4_per_token_head import reshape_and_cache_int4

    dev = key.device
    nb = T // block
    kc = torch.zeros(nb, block, H, D // 2, device=dev, dtype=torch.uint8)
    vc = torch.zeros_like(kc)
    ks = torch.zeros(nb, block, H, device=dev, dtype=torch.float32)
    vs = torch.zeros_like(ks)
    slots = torch.arange(T, device=dev, dtype=torch.int64)
    reshape_and_cache_int4(key, value, kc, vc, slots, k_scale_cache=ks, v_scale_cache=vs)
    return _unpack(kc, ks, T, H, D), _unpack(vc, vs, T, H, D)


def _report(stage, name, got, ref, tol_code, tol_zp, require_exact_scale):
    (q, zp, sc), (rq, rzp, rsc) = got, ref
    n = rq.numel()
    m = rzp.numel()
    code_bad = int((q != rq).sum())
    zp_bad = int((zp != rzp).sum())
    scale_bad = int((sc != rsc).sum())
    delta = (q - rq).abs()
    one_level = bool((delta[delta > 0] == 1).all()) if code_bad else True

    code_ok = code_bad / n <= tol_code and one_level
    zp_ok = zp_bad / m <= tol_zp
    scale_ok = (scale_bad == 0) if require_exact_scale else (scale_bad / m <= tol_zp)
    ok = code_ok and zp_ok and scale_ok

    print(
        f"[{stage}/{name}] codes {code_bad}/{n} ({code_bad / n:.2e}, "
        f"max|d|={int(delta.max()) if code_bad else 0}) {'OK' if code_ok else 'FAIL'} | "
        f"zp {zp_bad}/{m} {'OK' if zp_ok else 'FAIL'} | "
        f"scale {scale_bad}/{m} {'OK' if scale_ok else 'FAIL'}"
    )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device; this test compares against a Triton kernel")
        return 2

    import vllm.v1.attention.ops.int4_per_token_head as V

    import qat_kv.int4_ptkh_fake_quant as Q
    from qat_kv.int4_ptkh_fake_quant import Int4PTKHConfig, int4_ptkh_codes

    dev = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

    T, H, D, B = args.tokens, args.heads, args.head_dim, args.block_size
    if T % B:
        print(f"FAIL: tokens ({T}) must be a multiple of block_size ({B})")
        return 2

    key = torch.randn(T, H, D, device=dev, dtype=dtype)
    value = torch.randn(T, H, D, device=dev, dtype=dtype)
    cfg = Int4PTKHConfig(head_dim=D)
    ok = True

    # ---- strict: quantizer only ----------------------------------------
    # Pre-rotate with vLLM's own transform, then neutralize the rotation on
    # both sides so the two quantizers see byte-identical input.
    k_rot = V.single_rht(key.float()).to(dtype)
    v_rot = V.single_rht(value.float()).to(dtype)
    identity = lambda t, inverse=False, **kw: t  # noqa: E731

    v_orig, q_orig = V.single_rht, Q.rht
    V.single_rht, Q.rht = identity, identity
    try:
        ref_k, ref_v = _store(k_rot, v_rot, T, H, D, B)
        got_k = tuple(t.squeeze(-1) if t.dim() == 3 and t.shape[-1] == 1 else t
                      for t in int4_ptkh_codes(k_rot, cfg))
        got_v = tuple(t.squeeze(-1) if t.dim() == 3 and t.shape[-1] == 1 else t
                      for t in int4_ptkh_codes(v_rot, cfg))
    finally:
        V.single_rht, Q.rht = v_orig, q_orig

    ok &= _report("strict", "key", got_k, ref_k, STRICT_CODE_TOL, STRICT_ZP_TOL, True)
    ok &= _report("strict", "value", got_v, ref_v, STRICT_CODE_TOL, STRICT_ZP_TOL, True)

    # ---- full: including the rotation ----------------------------------
    ref_k, ref_v = _store(key, value, T, H, D, B)
    for name, src, ref in (("key", key, ref_k), ("value", value, ref_v)):
        rq, rzp, rsc = ref
        q, zp, sc = int4_ptkh_codes(src, cfg)
        ref_deq = (rq - rzp[..., None]) * rsc[..., None]
        got_deq = (q - zp) * sc
        rel = ((got_deq - ref_deq).norm() / ref_deq.norm()).item()
        good = rel <= FULL_DEQUANT_REL_TOL
        ok &= good
        print(f"[full/{name}] dequant rel {rel:.3e} (bound {FULL_DEQUANT_REL_TOL:.0e}) "
              f"{'OK' if good else 'FAIL'}")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
