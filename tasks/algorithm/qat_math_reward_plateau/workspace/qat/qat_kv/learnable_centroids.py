"""Learnable Lloyd-Max centroids for the TurboQuant key quantizer.

Why this exists
---------------
Every other knob in this repository was found to be blind to the quantization
residual.  The alignment loss ``d(f_q(x;theta), sg[f(x;theta)])`` does not fall
at any learning rate or coefficient, because a straight-through estimator sets
``d f_q / d theta ~= d f / d theta``, so the residual ``f_q - f`` is first-order
invariant in ``theta``.  Measured: the raw alignment loss sits in
[0.00053, 0.00065] across a 10x learning-rate range, a 10x coefficient range,
and 104 steps.

The centroids are different.  The reference forward runs with the quantizer
disabled, so it does not depend on them at all:

    d f / d centroids == 0      and      d f_q / d centroids != 0

The co-moving-target degeneracy therefore does not exist for these parameters,
and the gradient that reaches them is an exact gather, not an STE
approximation.  They are the one handle that can reduce the residual without
pushing ``theta`` away from the teacher it was initialised from.

Monotonicity
------------
vLLM's kernel picks a level with ``searchsorted(midpoints, y)`` and then reads
``centroids[idx]``.  That mapping is only coherent if the centroids are sorted,
so they are parameterised as a base plus a cumulative sum of positive gaps
rather than as free values that are re-sorted afterwards (re-sorting would make
the gradient discontinuous exactly when two levels cross).

Deployment
----------
vLLM stores the table per layer as a plain runtime tensor
(``layer._tq_centroids`` / ``layer._tq_midpoints``, set in
``turboquant_attn.py:_ensure_on_device``), so a learned table can be pushed to
the workers.  Training against a table that cannot be deployed would recreate
exactly the training/rollout mismatch this project spent its effort removing.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

__all__ = ["LearnableKeyCentroids", "inverse_softplus"]


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(exp(x) - 1)``."""
    return x + torch.log(-torch.expm1(-x))


class LearnableKeyCentroids(nn.Module):
    """A sorted, learnable centroid table initialised at Lloyd-Max.

    ``forward()`` returns a ``[n_levels]`` fp32 tensor that is sorted ascending
    by construction:

        c[0] = base
        c[i] = base + sum_{j<i} softplus(raw_gaps[j])

    Args:
        init: the Lloyd-Max table to start from, shape ``[n_levels]``.
    """

    def __init__(self, init: torch.Tensor):
        super().__init__()
        init = init.detach().to(torch.float32).flatten()
        if init.numel() < 2:
            raise ValueError(f"need at least 2 levels, got {init.numel()}")
        c0, _ = init.sort()
        gaps = c0[1:] - c0[:-1]
        if bool((gaps <= 0).any()):
            raise ValueError("initial centroids must be strictly increasing after sorting")
        self.base = nn.Parameter(c0[:1].clone())
        self.raw_gaps = nn.Parameter(inverse_softplus(gaps).clone())
        self.register_buffer("init_table", c0.clone(), persistent=True)

    @property
    def n_levels(self) -> int:
        return int(self.raw_gaps.numel()) + 1

    def forward(self) -> torch.Tensor:
        gaps = torch.nn.functional.softplus(self.raw_gaps)
        return torch.cat([self.base, self.base + torch.cumsum(gaps, dim=0)], dim=0)

    @torch.no_grad()
    def drift(self) -> dict:
        """How far the table has moved from Lloyd-Max, for logging."""
        c = self.forward()
        d = c - self.init_table
        return {
            "centroids/max_abs_drift": float(d.abs().max()),
            "centroids/mean_abs_drift": float(d.abs().mean()),
            "centroids/min_gap": float((c[1:] - c[:-1]).min()),
            "centroids/span": float(c[-1] - c[0]),
        }
