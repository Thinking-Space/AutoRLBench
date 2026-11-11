from __future__ import annotations

from .sample import Sample


def validate_messages(prompt: object) -> list[dict[str, str]]:
    if not isinstance(prompt, list):
        raise ValueError("native message input required")
    return prompt


def render_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"<{message['role']}>{message['content']}" for message in messages
    )


def run_generation(sample: Sample) -> Sample:
    messages = validate_messages(sample.prompt)
    prompt_text = render_messages(messages)

    # Training needs the rendered text. The bug is that the same public field
    # is later treated as the retry input representation without restoration.
    sample.prompt = prompt_text
    sample.tokens = [len(prompt_text), 1, 2, 3]
    sample.response = "example generated response"
    return sample
