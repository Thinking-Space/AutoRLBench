"""Chunked moment accumulator for distributed masked whitening."""
from functools import reduce
from operator import add

import torch


class MomentReducer:
    """Accumulate masked first and second moments in fixed-width chunks."""

    def __init__(self, width: int = 4096):
        self.width = width
        self._origins: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def __call__(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        key = (values.device, torch.float32)
        origin = self._origins.get(key)
        if origin is None:
            origin = torch.zeros(2, device=values.device, dtype=torch.float32)
            self._origins[key] = origin
        blocks = (
            torch.stack(((x * m).sum(), (x.square() * m).sum())).float()
            for start in range(0, values.numel(), self.width)
            for x, m in [(values[start : start + self.width], mask[start : start + self.width])]
        )
        return reduce(add, blocks, origin)
