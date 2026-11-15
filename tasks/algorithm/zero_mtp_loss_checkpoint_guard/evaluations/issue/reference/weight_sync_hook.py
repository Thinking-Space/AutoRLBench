"""Keep Qwen MTP weights while using base-model logits for zero-MTP-loss RL."""

import re
from copy import copy
from functools import wraps
from inspect import signature


def install_base_only_postprocess(model_class):
    original = model_class._postprocess
    if getattr(original, "_training_zero_mtp_loss", False):
        return
    forward_signature = signature(original)

    @wraps(original)
    def postprocess(self, *args, **kwargs):
        config = self.config
        call = forward_signature.bind(self, *args, **kwargs)
        if (not config.mtp_num_layers or config.mtp_loss_scaling_factor != 0.0
                or call.arguments.get("labels") is not None):
            return original(self, *args, **kwargs)

        # RL requests logits with no labels/position IDs. Only the postprocess
        # call sees MTP disabled; constructed modules and checkpoint config stay intact.
        base_config = copy(config)
        base_config.mtp_num_layers = None
        call.arguments["mtp_in_postprocess"] = False
        self.config = base_config
        try:
            return original(*call.args, **call.kwargs)
        finally:
            self.config = config

    postprocess._training_zero_mtp_loss = True
    model_class._postprocess = postprocess


def megatron_init(args):
    from miles.backends.megatron_utils.megatron_to_hf import qwen3_5
    from megatron.core.models.gpt.gpt_model import GPTModel

    install_base_only_postprocess(GPTModel)

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
