# SPDX-License-Identifier: Apache-2.0
"""Configuration for the simple min/max KV-cache quantizers.

Two rules share this module because they differ in exactly one axis:

    naive_*   keys and values both quantised per token, asymmetric min/max over
              head_dim (optionally sub-divided).  This is the LLM-QAT-style
              baseline that KIVI and TurboQuant are measured against.
    kivi_*    values per token as above; keys per *channel* over a group of
              tokens, which is KIVI's actual contribution.

Both may keep the most recent ``value_residual_length`` values in full
precision.  That tail is not a detail: measured on this model at 2 bits, the
windowed and windowless KIVI variants quantise keys byte-identically
(max|dK| = 0) and the entire 41.67 vs 28.33 AIME24 gap comes from the value
tail alone.  It is therefore a first-class config field rather than something
folded into the key rule.

Neither rule needs calibration -- every scale is computed from the tensor being
quantised -- which is why they can be dropped onto a plain BF16 checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# key_rule: "per_token" (naive baseline) or "per_channel" (KIVI).
# group_size is the token-group width for per_channel keys and the channel-group
# width for per-token quantisation; None means the whole head_dim.
SIMPLEKV_PRESETS: dict[str, dict] = {
    "naive_kv4": {"bits": 4, "key_rule": "per_token", "group_size": None,
                  "value_residual_length": 0},
    "naive_kv2": {"bits": 2, "key_rule": "per_token", "group_size": None,
                  "value_residual_length": 0},
    "naive_kv4_g32": {"bits": 4, "key_rule": "per_token", "group_size": 32,
                      "value_residual_length": 0},
    "naive_kv2_g32": {"bits": 2, "key_rule": "per_token", "group_size": 32,
                      "value_residual_length": 0},
    # Value-tail variants: same key rule, full-precision recent values.
    "naive_kv4_vw32": {"bits": 4, "key_rule": "per_token", "group_size": 32,
                       "value_residual_length": 32},
    "naive_kv2_vw32": {"bits": 2, "key_rule": "per_token", "group_size": 32,
                       "value_residual_length": 32},
    # KIVI proper.
    "kivi_kv4": {"bits": 4, "key_rule": "per_channel", "group_size": 32,
                 "value_residual_length": 32},
    "kivi_kv2": {"bits": 2, "key_rule": "per_channel", "group_size": 32,
                 "value_residual_length": 32},
    "kivi_kv2_nowin": {"bits": 2, "key_rule": "per_channel", "group_size": 32,
                       "value_residual_length": 0},
}


@dataclass(frozen=True)
class SimpleKVConfig:
    head_dim: int
    bits: int = 4
    key_rule: str = "per_token"
    group_size: int | None = None
    value_residual_length: int = 0

    def __post_init__(self) -> None:
        if self.bits not in (2, 3, 4, 8):
            raise ValueError(f"bits must be 2, 3, 4 or 8; got {self.bits}")
        if self.key_rule not in ("per_token", "per_channel"):
            raise ValueError(f"unknown key_rule {self.key_rule!r}")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group_size must be positive or None")
        if self.key_rule == "per_channel" and self.group_size is None:
            raise ValueError(
                "per_channel keys need an explicit token-group width; it must "
                "equal the paged block size so a group never straddles two "
                "blocks, which are not contiguous in memory"
            )
        if self.value_residual_length < 0:
            raise ValueError("value_residual_length must be non-negative")

    @property
    def value_group(self) -> int:
        return self.group_size or self.head_dim

    @property
    def key_data_bytes(self) -> int:
        return math.ceil(self.head_dim * self.bits / 8)

    @property
    def value_data_bytes(self) -> int:
        return math.ceil(self.head_dim * self.bits / 8)

    @property
    def value_meta_bytes(self) -> int:
        """fp16 scale + zero per channel group, per token."""
        return 4 * (self.head_dim // self.value_group)

    @property
    def key_meta_bytes_per_token(self) -> int:
        """Amortised key metadata.

        per_token keys carry their own scale/zero like values do.  per_channel
        keys share one scale/zero per channel across a whole token group, so the
        cost per token is head_dim * 4 / group_size -- at head_dim 128 and
        group 32 that is 16 bytes/token against TurboQuant's 2, which belongs
        in any memory claim rather than being quietly dropped.
        """
        if self.key_rule == "per_token":
            return 4 * (self.head_dim // self.value_group)
        return math.ceil(4 * self.head_dim / self.group_size)

    @property
    def slot_size(self) -> int:
        return (self.key_data_bytes + self.key_meta_bytes_per_token
                + self.value_data_bytes + self.value_meta_bytes)

    @property
    def slot_size_aligned(self) -> int:
        s = self.slot_size
        return s + (s % 2)

    @classmethod
    def from_cache_dtype(cls, cache_dtype: str, head_dim: int) -> "SimpleKVConfig":
        if cache_dtype not in SIMPLEKV_PRESETS:
            raise ValueError(
                f"unknown simplekv preset {cache_dtype!r}; "
                f"known: {', '.join(sorted(SIMPLEKV_PRESETS))}"
            )
        return cls(head_dim=head_dim, **SIMPLEKV_PRESETS[cache_dtype])
