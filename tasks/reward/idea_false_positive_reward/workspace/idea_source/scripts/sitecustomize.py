"""Process-wide compatibility shim for vLLM 0.11 + Transformers 5.x."""

from transformers import PreTrainedTokenizerBase


if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore[attr-defined]
        lambda tokenizer: tokenizer.all_special_tokens
    )
