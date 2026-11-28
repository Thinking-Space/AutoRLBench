"""Minimal reproduction of Slime's historical HF-to-Megatron load order."""


def load_hf_weights(model_chunks, source_bias):
    """Copy the FP32 HF correction bias into the existing model buffers."""
    for model_chunk in model_chunks:
        model_chunk.router.expert_bias.copy_(source_bias)
