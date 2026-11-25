"""Opt-in: replace the k2v3 key-dequant centroid GATHER with a select tree.

What this is fixing
-------------------
The stock stage-1 kernel
(`vllm/v1/attention/ops/triton_turboquant_decode.py`) dequantizes keys by
indexing the centroid table once per (KV token, head dim) element:

    c_vals = tl.load(Centroids_ptr + mse_idx, mask=..., other=0.0)

`mse_idx` is a [BLOCK_KV, BLOCK_D] tensor, so this is a 512-lane gather per
tile iteration at the default BLOCK_KV=4, D=128 -- against a table that for
2-bit MSE holds FOUR floats.  A gather cannot coalesce: it is one address
computation and one load instruction per lane, every iteration, and it sits
directly in the inner loop.

That this is the real cost is what the measurements say.  k2v3 moves ~6.6x
FEWER KV bytes than BF16 (2.4 bits/element vs 16) and yet decodes 11.2x
SLOWER (102.6 vs 9.18 ms/step at 30720 ctx, batch 16), so the kernel was
never bandwidth-bound.  The launch-config space is flat
(opqad/kernels/tq_decode_tune.py: BLOCK_KV/warps/stages, nothing beat the
defaults), NUM_KV_SPLITS is flat (102.6 / 102.8 / 103.2 at 32 / 64 / 128),
and the ported upstream v2 kernel -- which fixes the grid and adds tensor
cores but keeps a per-element table read -- is WORSE (147.9 ms).  A cost
that survives every tiling and parallelism change is a per-element
instruction cost, not a memory or occupancy one.

For MSE_BITS == 2 the table has 2**2 = 4 entries, so the gather is
replaceable by three `tl.where`s over four scalars held in registers.

Bitwise identity
----------------
Selecting among the same four fp32 values yields the same fp32 values the
gather returned; nothing downstream changes.  The one behavioural
difference is in MASKED lanes, where the gather substitutes 0.0 and the
select substitutes some centroid.  It does not reach the output: masked KV
rows are overwritten by `scores = tl.where(kv_mask, scores, -inf)` a few
lines later, and masked head dims are excluded by the `tl.where(d_mask,
...)` inside both reductions that consume `c_vals` (and BLOCK_D == D == 128
here, so there are no masked head dims at all).  verify_lut.py checks this
empirically rather than taking the argument on faith.

Implementation note
-------------------
The kernel body is source-transformed, not copied.  A 300-line copy of a
kernel from the shared install would silently rot the moment upstream
touched it; the transform asserts on its anchor text instead, so a changed
upstream fails loudly at apply() time.  The launcher is reused from
opqad/kernels/tq_decode_tune.py for the same reason.

Default off
-----------
QAT_TQ_DECODE_LUT=1.  Mutually exclusive with QAT_TQ_DECODE_TUNE and
QAT_TQ_DECODE_V2 (checked in opqad/integrations/vllm_plugin.py): all three rebind the
same decode entry point.

Every variant here is a measured negative, and they are kept as one
-------------------------------------------------------------------
    transform   loop SASS   measured
    stock             622   102.6 ms
    onebyte           622   104.3      (identical count: ptxas already
                                        removed the load by itself)
    dedup             629   -- rejected on count, see below
    select            746   127.0
    both                -   128.2

Instruction count predicts wall clock on this kernel, which is the whole
point of the roofline finding (experiments/2026-09-22/tq_decode_tune/
kernel_sass.py): it runs at 71% of its issue ceiling, so what it issues is
what it costs.  `dedup` was rejected on its count alone rather than being
given a bench slot -- a transform that issues MORE cannot be faster here.

Why there is no `word` transform here
-------------------------------------
The obvious next step after `dedup` is to stop loading bytes at all: the
loop issues 54 `LDG.E.U8`, one byte and one 64-bit address apiece, and a
`u32` load would serve four of them.  Two independent facts close it.

First, it is illegal.  `slot_size = key_packed_size(34) + value_packed_size
(52) = 86`, and `slot_size_aligned` (quantization/turboquant/config.py:177)
rounds up to EVEN, not to 4.  Every cache stride is a multiple of 86, so a
slot base is 2-byte aligned and 4-byte aligned only on even slot indices.
Making a u32 load legal means padding the slot to 88 -- a KV cache FORMAT
change (store kernel, and effective_head_size = slot//2 = 44), not a perf
knob.

Second, it would not pay even if it were legal.  `dedup` removes a 4x
redundant byte gather and the loads fall only 71 -> 68, not the ~12 per lane
the arithmetic predicts -- ptxas and the LSU were already collapsing those
redundant byte loads.  So a wider load buys back almost no issue slots while
paying a register fan-out cost that is 16x for u32 against dedup's 4x, and
dedup already lost on exactly that term (moves 46 -> 58).
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import tempfile
import textwrap

ENV_FLAG = "QAT_TQ_DECODE_LUT"
# Which source transforms to apply.  Separate so each can be attributed:
# "select" alone MEASURED 127.0 ms vs the stock 102.6, i.e. worse.
# "dedup" rewrites the same block "onebyte" does, so the two are
# alternatives rather than a combination; "both" stays select+onebyte.
_MODES = ("select", "onebyte", "dedup", "both")

_applied = False

# Anchor A: the gather itself, replaced by the select tree.
_GATHER_SRC = """            # Centroid gather + dot product
            c_vals = tl.load(
                Centroids_ptr + mse_idx,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
"""

_SELECT_SRC = """            if MSE_BITS == 2:
                c_vals = tl.where(
                    mse_idx == 0, _c_lut0,
                    tl.where(mse_idx == 1, _c_lut1,
                             tl.where(mse_idx == 2, _c_lut2, _c_lut3)))
            else:
                c_vals = tl.load(
                    Centroids_ptr + mse_idx,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
"""

# Anchor C: the two-byte stitch used to read the packed MSE index.  A
# 2-bit (or 4-bit) field divides 8 evenly, so it never straddles a byte and
# the high byte contributes nothing after the shift and mask.  The stock
# kernel loads it anyway because the same code also serves 3-bit values,
# which DO straddle -- note that the VALUE path already has exactly this
# specialization ("a single load suffices and no 16-bit stitching is
# needed") for VQB == 2, and the key path simply never got it.
#
# Unlike the select tree this REMOVES work rather than moving it: one of
# every two byte loads, plus the shift and the or, per lane per tile.
#
# It also measured flat (104.3 vs 102.6 ms), and job 123119 says why: the
# transformed kernel compiles to 622 loop instructions, exactly the stock
# count.  ptxas had already dead-code-eliminated the second load on its own.
_STITCH_SRC = """            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask
"""

_ONEBYTE_SRC = """            if MSE_BITS == 2 or MSE_BITS == 4:
                mse_idx = (mse_raw0 >> mse_bit_shift[None, :]) & mse_mask
            else:
                mse_raw1 = tl.load(
                    KV_cache_ptr + mse_addrs0 + 1,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                raw16 = mse_raw0 | (mse_raw1 << 8)
                mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask
"""

# Anchor B: hoist the four scalar loads above the tile loop.  Triton would
# very likely hoist them anyway, but "very likely" is not a thing to leave
# in the inner loop of the kernel this module exists to speed up.
_HOIST_ANCHOR = """    # Online softmax accumulators
"""

_HOIST_SRC = """    if (not KEY_FP8) and MSE_BITS == 2:
        _c_lut0 = tl.load(Centroids_ptr + 0)
        _c_lut1 = tl.load(Centroids_ptr + 1)
        _c_lut2 = tl.load(Centroids_ptr + 2)
        _c_lut3 = tl.load(Centroids_ptr + 3)

""" + _HOIST_ANCHOR


# Anchor D: the whole MSE unpack, replaced by a deduplicated byte load.
#
# At MSE_BITS == 2 the stock code indexes the cache with a
# [BLOCK_KV, BLOCK_D] byte address tensor whose column index is
# `d_offs // 4` -- so 128 head dims address only 32 distinct bytes and each
# one is fetched FOUR times, with four separate 64-bit address computations.
#
# MEASURED WORSE, and rejected on instruction count rather than wall clock
# (job 123119): the loop body goes 622 -> 629 SASS instructions.  The loads
# do fall, 71 -> 68, but Triton materializes the broadcast fan-out with
# explicit register moves, 46 -> 58, and that costs more than the loads it
# saved.  Kept here because the negative is the point: on a kernel already
# at 71% of its issue roofline, moving work from the memory pipe into the
# register file is not a win, it is a rename.
_MSEBLOCK_SRC = """            # MSE unpack + norms
            mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(
                KV_cache_ptr + mse_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask
"""

_DEDUP_SRC = """            # MSE unpack + norms
            if MSE_BITS == 2:
                mse_packed = tl.load(
                    KV_cache_ptr
                    + slot_bases[:, None] + _mse_uniq[None, :],
                    mask=kv_mask[:, None] & _mse_uniq_mask[None, :],
                    other=0,
                ).to(tl.int32)
                mse_idx = (
                    tl.reshape(
                        tl.broadcast_to(
                            mse_packed[:, :, None],
                            (BLOCK_KV, BLOCK_D // 4, 4),
                        ),
                        (BLOCK_KV, BLOCK_D),
                    )
                    >> mse_bit_shift[None, :]
                ) & mse_mask
            else:
                mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
                mse_raw0 = tl.load(
                    KV_cache_ptr + mse_addrs0,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                mse_raw1 = tl.load(
                    KV_cache_ptr + mse_addrs0 + 1,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                raw16 = mse_raw0 | (mse_raw1 << 8)
                mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask
"""

# The unique-byte index vector, hoisted beside the stock loop-invariants.
# `_mse_uniq_mask` is what keeps this honest when HEAD_DIM does not fill
# BLOCK_D: the stock gather masked on `d_mask`, and the corresponding mask
# on the deduplicated axis is "this byte is inside MSE_BYTES".
_DEDUP_HOIST_SRC = """    if (not KEY_FP8) and MSE_BITS == 2:
        _mse_uniq = tl.arange(0, BLOCK_D // 4)
        _mse_uniq_mask = _mse_uniq < MSE_BYTES

""" + _HOIST_ANCHOR


def mode() -> str:
    """"" (off) or one of _MODES.  Raises on anything else rather than
    quietly falling back to a transform the caller did not ask for."""
    raw = os.environ.get(ENV_FLAG, "").strip()
    if raw in ("", "0", "false", "False"):
        return ""
    if raw == "1":
        raw = "both"
    if raw not in _MODES:
        raise ValueError(
            f"{ENV_FLAG}={raw!r}; expected one of {_MODES} (or 1 = both)"
        )
    return raw


def enabled() -> bool:
    return mode() != ""


def build_kernel(which: str = "both"):
    """Return a `triton.jit` stage-1 kernel with `which` transform applied.

    Raises if the installed kernel's source no longer contains the anchors
    this transform edits -- that means upstream changed the dequant path and
    the replacement needs to be re-derived, not silently skipped.
    """
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _tq_decode_stage1,
    )

    src = textwrap.dedent(inspect.getsource(_tq_decode_stage1.fn))
    # getsource includes the @triton.jit line; the generated module re-adds
    # its own, so strip it here rather than emitting it twice.
    src = src[src.index("def _tq_decode_stage1("):]
    anchors = []
    if which in ("select", "both"):
        anchors += [_GATHER_SRC, _HOIST_ANCHOR]
    if which in ("onebyte", "both"):
        anchors.append(_STITCH_SRC)
    if which == "dedup":
        anchors += [_MSEBLOCK_SRC, _HOIST_ANCHOR]
    for anchor in anchors:
        if anchor not in src:
            raise RuntimeError(
                "tq_decode_lut: anchor not found in the installed "
                "_tq_decode_stage1 source; upstream changed the key dequant "
                "path and this transform must be re-derived:\n"
                + anchor
            )
    if which in ("select", "both"):
        src = src.replace(_GATHER_SRC, _SELECT_SRC, 1)
        src = src.replace(_HOIST_ANCHOR, _HOIST_SRC, 1)
    if which in ("onebyte", "both"):
        src = src.replace(_STITCH_SRC, _ONEBYTE_SRC, 1)
    if which == "dedup":
        src = src.replace(_MSEBLOCK_SRC, _DEDUP_SRC, 1)
        src = src.replace(_HOIST_ANCHOR, _DEDUP_HOIST_SRC, 1)
    src = src.replace("def _tq_decode_stage1(", "def _tq_decode_stage1_lut(", 1)
    src = ("from vllm.triton_utils import tl, triton\n\n\n@triton.jit\n" + src)

    # Triton reads a jit function's source off disk (`@jit functions should
    # be defined in a Python file`), so exec-ing this into a namespace does
    # not work -- it has to be a real module.  The filename is keyed on a
    # hash of the source so concurrent processes converge on one file and a
    # changed upstream kernel cannot reuse a stale one.
    digest = hashlib.sha256(src.encode()).hexdigest()[:16]
    tmpdir = os.environ.get("TMPDIR") or tempfile.gettempdir()
    path = os.path.join(tmpdir, f"qat_tq_decode_lut_{digest}.py")
    if not os.path.exists(path):
        tmp = f"{path}.{os.getpid()}"
        with open(tmp, "w") as fh:
            fh.write(src)
        os.replace(tmp, path)

    spec = importlib.util.spec_from_file_location(
        f"qat_tq_decode_lut_{digest}", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._tq_decode_stage1_lut


def apply(verbose: bool = True) -> bool:
    """Patch the in-process k2v3 decode kernel with the select-tree dequant.

    Returns True if applied.  Safe to call when vLLM has no TurboQuant
    backend loaded (an OSCAR or BF16 arm): the import fails and this returns
    False rather than raising.  Must be called before the first decode call.
    """
    global _applied
    if _applied or not enabled():
        return False
    try:
        import vllm.v1.attention.backends.turboquant_attn as _tqa

        from opqad.kernels.tq_decode_tune import build_launcher
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[tq-lut] not applied: {type(exc).__name__}: {exc}",
                  flush=True)
        return False

    which = mode()
    _tqa.triton_turboquant_decode_attention = build_launcher(
        stage1_kernel=build_kernel(which)
    )
    _applied = True
    if verbose:
        print(f"[tq-lut] k2v3 key dequant transform={which} "
              "(in-process patch; shared install untouched)", flush=True)
    return True
