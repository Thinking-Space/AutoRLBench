"""miles_plugins HuggingfaceAttention — defective version.

This is a minimal stub reproducing the signature that causes the TypeError
when megatron 0.18 calls build_module(..., name="self_attention", ...).

The real miles_plugins/models/hf_attention.py has the same __init__ signature:
no `name` parameter, no **kwargs.
"""


class HuggingfaceAttention:
    """Stub of miles_plugins HuggingfaceAttention (defective version).

    megatron 0.18 calls:
        build_module(spec, config=config, layer_number=layer_number, name="self_attention")

    This class has no `name` parameter, so the call raises:
        TypeError: HuggingfaceAttention.__init__() got an unexpected keyword argument 'name'
    """

    def __init__(self, config, layer_number=None):
        self.config = config
        self.layer_number = layer_number
        self.initialized = True
