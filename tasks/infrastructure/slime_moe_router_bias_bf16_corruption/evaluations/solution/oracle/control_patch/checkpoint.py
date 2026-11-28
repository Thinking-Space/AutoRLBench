"""Repaired Slime HF-to-Megatron load order."""


def load_hf_weights(model_chunks, source_bias):
    """Restore FP32 correction buffers before copying FP32 HF values."""
    for model_chunk in model_chunks:
        for module in model_chunk.modules():
            if hasattr(module, "_maintain_float32_expert_bias"):
                module._maintain_float32_expert_bias()

    for model_chunk in model_chunks:
        model_chunk.router.expert_bias.copy_(source_bias)
