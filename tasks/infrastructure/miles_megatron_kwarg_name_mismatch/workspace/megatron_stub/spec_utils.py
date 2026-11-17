"""Stub for megatron 0.18 build_module / spec_utils.

This minimal stub reproduces the calling convention that triggers the TypeError:
  build_module(spec, config=config, layer_number=layer_number, name="self_attention")
"""

class ModuleSpec:
    def __init__(self, module=None, **kwargs):
        self.module = module


def build_module(module_spec, *args, **kwargs):
    """megatron 0.18 build_module — passes name= to the target __init__."""
    if isinstance(module_spec, ModuleSpec) and module_spec.module is not None:
        target = module_spec.module
    elif isinstance(module_spec, type):
        target = module_spec
    else:
        raise TypeError(f"Cannot build module from {module_spec!r}")
    return target(*args, **kwargs)
