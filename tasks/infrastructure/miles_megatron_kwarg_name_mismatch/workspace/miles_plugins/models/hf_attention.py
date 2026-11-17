"""miles_plugins.models.hf_attention stub (defective version).

Reproduces the signature that triggers TypeError when megatron 0.18
calls build_module(..., name="self_attention", ...).
"""

class HuggingfaceAttention:
    """Defective stub: __init__ has no `name` parameter, no **kwargs.

    megatron 0.18 calls:
        build_module(spec, config=config, layer_number=layer_number, name="self_attention")
    -> TypeError: HuggingfaceAttention.__init__() got an unexpected keyword argument 'name'
    """

    def __init__(self, config, layer_number=None):
        self.config = config
        self.layer_number = layer_number
