"""Minimal reproduction of the megatron 0.18 + miles_plugins name= kwarg TypeError.

Run this with:
    python3 reproduce.py

Expected output with the defect:
    TypeError: HuggingfaceAttention.__init__() got an unexpected keyword argument 'name'

The defect is in the interaction between:
  - megatron 0.18: transformer_layer.py:363 calls
      build_module(spec, config=config, layer_number=layer_number, name="self_attention")
  - miles_plugins: HuggingfaceAttention.__init__(self, config, layer_number=None)
    has no `name` parameter.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "megatron_stub"))
sys.path.insert(0, os.path.dirname(__file__))

from megatron_stub.transformer_layer import simulate_build_module_call
from miles_plugins.models.hf_attention import HuggingfaceAttention
from megatron_stub.spec_utils import ModuleSpec

spec = ModuleSpec(module=HuggingfaceAttention)

print("Simulating megatron 0.18 build_module(..., name='self_attention') call ...")
try:
    obj = simulate_build_module_call(spec, config=None, layer_number=1,
                                     name="self_attention")
    print("ERROR: expected TypeError was not raised — defect not reproduced.")
    sys.exit(1)
except TypeError as e:
    print(f"TypeError raised (expected): {e}")
    print("Defect confirmed: megatron 0.18 name= kwarg rejected by HuggingfaceAttention.")
    sys.exit(0)
