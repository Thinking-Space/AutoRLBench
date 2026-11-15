"""Adapt modern Megatron MTP names to the pinned Qwen3.5 converter."""

import re
from functools import wraps


def megatron_init(args):
    from miles.backends.megatron_utils.megatron_to_hf import qwen3_5

    original = qwen3_5._convert_mtp_layer
    if getattr(original, "_training_modern_mtp_names", False):
        return

    @wraps(original)
    def convert_mtp_layer(args, name, param, layer_idx):
        # Modern Megatron renamed this child without changing its parameters.
        name = re.sub(r"(\.mtp\.layers\.\d+\.)mtp_model_layer\.", r"\1transformer_layer.", name)
        return original(args, name, param, layer_idx)

    convert_mtp_layer._training_modern_mtp_names = True
    qwen3_5._convert_mtp_layer = convert_mtp_layer
