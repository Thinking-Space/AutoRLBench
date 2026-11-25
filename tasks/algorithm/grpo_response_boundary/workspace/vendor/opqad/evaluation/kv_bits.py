"""Effective KV-cache bit rates, computed rather than hardcoded.

Why this exists
---------------
Every KV-quantization table in this repo quoted OSCAR at "2.14 bits" and KIVI at
"2.01".  Both were wrong, in the same direction, for two independent reasons:

1. **The scale/zero metadata was never counted.**  ``triton_oscar_store.py``
   stores an fp16 ``(scale, zero)`` pair per ``(token, head)`` vector, and
   ``OscarConfig.meta_bytes`` is ``num_groups * 4``.  At ``head_dim=128`` and
   ``group_size=128`` that is 32 bits spread over 128 elements: **+0.25 bits per
   element**, before any window.  A 2-bit scheme is never 2.00 bits.

2. **Four layers ran in BF16.**  ``vllm/engine/arg_utils.py:1724`` unions
   ``OscarConfig.get_boundary_skip_layers`` into ``kv_cache_dtype_skip_layers``
   for any ``oscar_*`` or ``turboquant_*`` dtype, protecting ``{0, 1, L-2,
   L-1}``.  ``opqad/runtime/oscar_layer_skip_patch.py`` exists to undo it -- the paper
   quantizes every layer -- but it is opt-in and was never enabled on a landed
   run (job 61059: ``applied: v-stride-fix, per-request-staging``).

The two errors compound, and neither touches the KIVI or hybrid rows, which hook
the attention forward on a BF16 cache and have no skip list.  So the published
comparison understated exactly the arms it was meant to defend.  Computing the
rate from the configuration, in one place used by both the table and the result
JSON, is what keeps that from recurring.

What is counted
---------------
Per element of one KV tensor, averaged over layers::

    quantized element  = bits + meta_bits / group_elements
    windowed element   = 16                     (BF16 sink / recent / residual)
    skipped layer      = 16                     (whole layer in BF16)

``group_elements`` is the number of values sharing one ``(scale, zero)`` pair,
whichever axis they lie along -- 128 channels for OSCAR's per-token grid, 32
tokens for KIVI's per-channel key grid.  The axis matters enormously for
accuracy (see ``oscar-k-axis-is-the-weak-leg``) and not at all for the bit rate;
only the count does.

What is NOT counted, and must be said out loud wherever these numbers appear:
the KIVI and hybrid arms are *fake quantization into a BF16 cache*.  Their rate
is what the scheme would cost if it were stored, not what it costs today.  The
OSCAR rate is real storage.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Scheme", "OSCAR_INT2", "KIVI_2BIT", "HYBRID_2BIT", "BF16",
           "TQ_2BIT", "TQ_K2V3", "TQ_4BIT", "BY_NAME", "turboquant",
           "boundary_skip_count"]

BF16_BITS = 16.0
# fp16 scale + fp16 zero, as stored by triton_oscar_store.py and by KIVI's
# reference implementation.
META_BITS_PER_GROUP = 32.0


@dataclass(frozen=True)
class Scheme:
    """A KV-quantization scheme's storage cost.

    K and V are carried separately because every scheme here treats them
    differently -- that asymmetry is the whole subject -- and averaged at the
    end, since a cache holds one of each per position.
    """

    name: str
    k_bits: float
    v_bits: float
    k_group_elements: int
    v_group_elements: int
    # Bits of side information stored per group.  32 = an fp16 (scale, zero)
    # pair, which is what OSCAR's store kernel and KIVI's reference both write.
    # turboquant's KEY path is the exception: its Lloyd-Max codebook needs only
    # a 2-byte fp16 ``vec_norm``, so 16.
    k_meta_bits: float = META_BITS_PER_GROUP
    v_meta_bits: float = META_BITS_PER_GROUP
    # Positions per sequence held in BF16.  OSCAR: a sink and a recent window.
    # KIVI: no window, but the newest unfilled group is a full-precision
    # residual, mean G/2 tokens.
    sink_tokens: int = 0
    recent_tokens: int = 0
    residual_tokens: float = 0.0
    # True when the arm is a hook over a BF16 cache, so the rate is what the
    # scheme would cost rather than what it costs.
    fake_quant: bool = False

    def _axis_bits(self, bits: float, group: int, meta: float) -> float:
        if group <= 0:
            raise ValueError(f"{self.name}: group_elements must be positive, "
                             f"got {group}")
        return bits + meta / group

    @property
    def quantized_bits(self) -> float:
        """Bits per element in the quantized region, K and V averaged."""
        k = self._axis_bits(self.k_bits, self.k_group_elements, self.k_meta_bits)
        v = self._axis_bits(self.v_bits, self.v_group_elements, self.v_meta_bits)
        return (k + v) / 2.0

    @property
    def bf16_tokens(self) -> float:
        return self.sink_tokens + self.recent_tokens + self.residual_tokens

    def layer_bits(self, context_len: int) -> float:
        """Bits per element for one quantized layer at this context length.

        The window is a fixed number of tokens, so its share shrinks as the
        context grows: quoting a rate without a context is meaningless.
        """
        if context_len <= 0:
            raise ValueError(f"context_len must be positive, got {context_len}")
        q = self.quantized_bits
        # A sequence shorter than the window is entirely BF16.  Clamping here
        # rather than letting the fraction exceed 1 keeps short-context smoke
        # runs from reporting a rate above 16.
        frac = min(self.bf16_tokens / context_len, 1.0)
        return q + frac * (BF16_BITS - q)

    def effective_bits(
        self,
        context_len: int,
        num_layers: int,
        skip_layers: int = 0,
    ) -> float:
        """Bits per element averaged over the whole cache.

        ``skip_layers`` is the count of layers held in BF16 by
        ``kv_cache_dtype_skip_layers`` -- 4 for the stock OSCAR and turboquant
        paths, 0 once ``VLLM_OSCAR_QUANT_ALL_LAYERS=1`` is set, and 0 for any
        fake-quant arm.
        """
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 0 <= skip_layers <= num_layers:
            raise ValueError(
                f"skip_layers {skip_layers} outside [0, {num_layers}]")
        quantized = num_layers - skip_layers
        total = quantized * self.layer_bits(context_len) + skip_layers * BF16_BITS
        return total / num_layers


def boundary_skip_count(num_layers: int, n: int = 2) -> int:
    """How many layers upstream protects, mirroring OSCAR's own rule.

    ``OscarConfig.get_boundary_skip_layers`` keeps the first and last ``n``
    attention layers in native dtype and clamps ``n`` to ``num_layers // 2``.
    Reproduced here (rather than imported) so the table can be computed without
    a vLLM install; ``tests/test_kv_bits.py`` pins it against the upstream
    source.
    """
    if n <= 0 or num_layers <= 0:
        return 0
    n = min(n, num_layers // 2)
    return len(sorted(set(list(range(n)) + list(range(num_layers - n, num_layers)))))


# OSCAR INT2 as the vLLM port implements it: one fp16 (scale, zero) per
# (token, head) vector at head_dim 128, group_size 128 -> 2.25 bits/element,
# plus a BF16 sink 64 + recent 256 per request.
OSCAR_INT2 = Scheme(
    name="oscar_int2",
    k_bits=2.0, v_bits=2.0,
    k_group_elements=128, v_group_elements=128,
    sink_tokens=64, recent_tokens=256,
)

# KIVI: keys share a scale down a CHANNEL across G=32 tokens, values across 32
# channels of one token.  Same 32 elements per group either way.  No window;
# the newest unfilled group is a full-precision residual, mean G/2 = 16 tokens.
KIVI_2BIT = Scheme(
    name="kivi_2bit",
    k_bits=2.0, v_bits=2.0,
    k_group_elements=32, v_group_elements=32,
    residual_tokens=16.0,
    fake_quant=True,
)

# The hybrid: KIVI's per-channel key grid over one paged block (32 tokens at
# VLLM_OSCAR_BLOCK_SIZE=32), OSCAR's rotated per-token value grid, OSCAR's
# window.  Cheaper than either parent at 32k, which the writeup has to say.
HYBRID_2BIT = Scheme(
    name="hybrid_2bit",
    k_bits=2.0, v_bits=2.0,
    k_group_elements=32, v_group_elements=128,
    sink_tokens=64, recent_tokens=256,
    fake_quant=True,
)

BF16 = Scheme(
    name="bf16",
    k_bits=BF16_BITS, v_bits=BF16_BITS,
    # One "group" the size of a vector carrying no metadata: an unquantized
    # cache stores no scale or zero.  Expressed as a huge group rather than a
    # special case so `quantized_bits` stays a single expression.
    k_group_elements=10**9, v_group_elements=10**9,
)


def turboquant(name: str, k_bits: float, v_bits: float) -> Scheme:
    """A turboquant preset, from ``TurboQuantConfig``'s own packed sizes.

    Keys carry a 2-byte fp16 ``vec_norm`` beside their Lloyd-Max indices;
    values carry the usual 4-byte (scale, zero).  turboquant quantizes every
    position -- no sink, no recent window, no residual -- which is the whole
    reason its 2-bit tier collapses where OSCAR's does not.
    """
    return Scheme(
        name=name,
        k_bits=k_bits, v_bits=v_bits,
        k_group_elements=128, v_group_elements=128,
        k_meta_bits=16.0, v_meta_bits=32.0,
    )


TQ_2BIT = turboquant("turboquant_2bit_nc", 2.0, 2.0)      # 2.1875
TQ_K2V3 = turboquant("turboquant_k2v3_nc", 2.0, 3.0)      # 2.6875
TQ_4BIT = turboquant("turboquant_4bit_nc", 4.0, 4.0)      # 4.1875

# Every scheme by the name the result JSON files it under, so a table can look
# one up instead of hardcoding a rate beside a path.
BY_NAME = {
    "bf16": BF16,
    "oscar_int2": OSCAR_INT2,
    "kivi_2bit": KIVI_2BIT,
    "hybrid_2bit": HYBRID_2BIT,
    "turboquant_2bit": TQ_2BIT,
    "turboquant_k2v3": TQ_K2V3,
    "turboquant_4bit": TQ_4BIT,
}
