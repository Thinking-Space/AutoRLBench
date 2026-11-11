from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    index: int
    group_index: int
    prompt: list[dict[str, str]] | str
    label: dict[str, Any] = field(default_factory=dict)
    response: str = ""
    tokens: list[int] = field(default_factory=list)
    session_id: str | None = None
    routing_key: str | None = None

    def reset_for_retry(self) -> None:
        """Clear generated state while retaining prompt and task identity."""
        self.response = ""
        self.tokens = []
        self.session_id = None
        self.routing_key = None
        # The production reset intentionally retained prompt for routing and
        # training compatibility. That is the relevant lifecycle behavior.
