"""Quantize EVERY attention layer, the way the OSCAR paper does.

The shared vLLM port keeps layers {0, 1, N-2, N-1} in native dtype
(``OscarConfig.get_boundary_skip_layers``, config.py).  Its own docstring says
this is a stand-in inherited from the TurboQuant integration, "until the
allocator-level" sink/recent window lands -- but that window has landed (the
staging arena, and now the per-request staging patch), and the paper
(arXiv:2605.17757) quantizes ALL layers: its ablation explicitly finds that
leaving boundary layers unquantized is unnecessary.  So the skip list is a
deviation from the paper, in both directions:

  * it spends KV memory the paper does not spend (4/36 layers at BF16 on
    Qwen3-8B is +11% over the paper's 2.28 BPE), and
  * it is not what the published rotations were tuned against -- upstream's
    zoo shipped with all layers quantized in SGLang.

**Unconditional since 2026-09-17.**  This used to be opt-in on
``VLLM_OSCAR_QUANT_ALL_LAYERS``, and being opt-in is what made every landed
OSCAR bit rate wrong (quoted 2.14, actually 4.33 on the 1.5B), killed job 63401
when the flag reached one side only, and put three full training arms
(63686/63687/63817) at roughly double the bit budget of the baselines they were
meant to beat.  The layer set is a property of the quantizer, not an
experimental variable, so there is nothing left to decide here.

``opqad/quantization/turboquant_fake_quant.oscar_quant_all_layers`` still *reads* the old
variable, and only to raise on a stale ``=0`` -- see its docstring.

The override is decided in the parent process (``EngineArgs.create_engine_config``
runs there and serialises ``cache_config.kv_cache_dtype_skip_layers`` into the
EngineCore), so unlike the kernel patches it does not depend on the
fork-vs-spawn question -- but the plugin still applies it everywhere so a log
line appears where the decision was taken.
"""

from __future__ import annotations

import os

__all__ = ["maybe_apply"]

_DONE = False


def maybe_apply(verbose: bool = True) -> bool:
    global _DONE
    if _DONE:
        return True

    # Raises on a stale VLLM_OSCAR_QUANT_ALL_LAYERS=0 rather than honouring it.
    # Importing here keeps the vLLM plugin path free of a qat_kv import at
    # module scope, which is what the rest of this package does too.
    from opqad.quantization.turboquant_fake_quant import oscar_quant_all_layers

    oscar_quant_all_layers()

    from vllm.model_executor.layers.quantization.oscar import config as oc

    def patched(model_config, n_unused: int = 2):
        if verbose:
            L = model_config.hf_text_config.num_hidden_layers
            print(f"[oscar-layers] pid={os.getpid()} boundary skip list -> NONE "
                  f"(all {L} layers quantized)", flush=True)
        return []

    oc.OscarConfig.get_boundary_skip_layers = staticmethod(patched)
    _DONE = True
    return True
