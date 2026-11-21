"""Small, model-agnostic KV activation fake-quantization helpers.

The package deliberately contains only *fake* quantization: tensors stay in
their original floating-point dtype and no cache is packed.  This is the
training-time behaviour used by the LLM-QAT reference implementation.
"""

from .fake_quant import (
    KVFakeQuantConfig,
    KVFakeQuantHook,
    KVFakeQuantizer,
    attach_kv_fake_quant_hooks,
    symmetric_fake_quant,
)

# KIVI is imported lazily by the rollout worker because its official CUDA
# extension is optional for ordinary LLM-QAT utilities and CPU tests.
from .kivi_qwen2 import (
    KIVI_COMMIT,
    KIVICache,
    KIVIQwen2Config,
    install_kivi_hf_rollout,
    load_official_kivi_kernels,
)

__all__ = [
    "KVFakeQuantConfig",
    "KVFakeQuantHook",
    "KVFakeQuantizer",
    "attach_kv_fake_quant_hooks",
    "symmetric_fake_quant",
    "KIVI_COMMIT",
    "KIVICache",
    "KIVIQwen2Config",
    "install_kivi_hf_rollout",
    "load_official_kivi_kernels",
]
