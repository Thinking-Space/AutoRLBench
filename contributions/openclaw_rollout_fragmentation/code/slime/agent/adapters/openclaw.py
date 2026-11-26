"""OpenClaw-specific policy for the OpenAI Chat Completions wire protocol."""

from __future__ import annotations

from slime.agent.adapters.openai import OpenAIAdapter


class OpenClawAdapter(OpenAIAdapter):
    """Preserve the assistant shape that OpenClaw replays on later turns."""

    log_prefix = "openclaw_adapter"

    def __init__(self, **kwargs) -> None:
        super().__init__(
            preserve_reasoning_in_manager=True,
            preserve_parallel_tool_calls=True,
            preserve_content_with_tool_calls=True,
            sort_tool_arguments=False,
            **kwargs,
        )


__all__ = ["OpenClawAdapter"]
