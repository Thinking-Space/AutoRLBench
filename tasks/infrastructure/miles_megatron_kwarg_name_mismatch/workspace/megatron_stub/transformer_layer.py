"""Stub for megatron 0.18 transformer_layer.build_module call.

This minimal stub reproduces the call that triggers the TypeError.
"""
from megatron_stub.spec_utils import build_module, ModuleSpec

def simulate_build_module_call(spec, **kwargs):
    """Simulates megatron 0.18 transformer_layer.py:363:
        self_attention = build_module(spec, config=config,
                                      layer_number=layer_number,
                                      name="self_attention")
    """
    return build_module(spec, **kwargs)
