from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    index: int
    group_index: int
    prompt: str
    label: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    reward: float | None = None
    remove_sample: bool = False
