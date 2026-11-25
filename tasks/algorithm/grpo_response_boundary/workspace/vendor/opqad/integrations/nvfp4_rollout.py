"""Re-quantize BF16 actor weights into a live vLLM NVFP4 engine, every step.

The problem
-----------
On-policy QAT needs the sampler to make the same errors the student is trained
to correct.  The actor carries NVFP4 fake-quant (``opqad.quantization.wa_scheme``) but the
rollout engine serves whatever weights it was given, and verl's weight sync
hands it BF16.  Left alone, the student would learn to compensate for a
quantization error its own sampler never commits.

Why vLLM's own loader cannot be reused per step
-----------------------------------------------
``ModelOptNvFp4LinearMethod.process_weights_after_loading`` is a one-shot:
it derives ``input_global_scale`` / ``weight_global_scale`` / ``alpha`` and then
``del``s the ``input_scale`` and ``weight_scale_2`` parameters it read them
from.  A second ``load_weights`` for those names would fail.

What makes a per-step path possible
-----------------------------------
Those deleted tensors are exactly the ones that *do not change*.  The per-tensor
amax is calibrated once by ``tools/analysis/ptq_nvfp4.py`` and frozen for the whole run
-- the same frozen amax the actor's fake-quant restores -- so
``weight_global_scale``, ``input_global_scale`` and ``alpha`` are constants.
Only the packed weight and its per-16-value block scale follow the weights.

And the kernel's post-processing turns out to be two pure functions:

    weight_scale <- swizzle_blockscale(raw_block_scale)
    weight       <- pad_nvfp4_weight_for_cutlass(packed_fp4)

so the final layout can be recomputed from scratch instead of mutated in place.

Engine setup this assumes
-------------------------
The engine must be built from the NVFP4 checkpoint directory with a real
``load_format`` (not ``dummy``).  Under ``dummy`` the ``input_scale`` and
``weight_scale_2`` parameters are random, and the ``alpha`` derived from them --
which this module never rewrites -- would be meaningless.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

import torch

LOGGER = logging.getLogger(__name__)

# vLLM fuses these HF projections into one linear.  Order matters: it is the
# concatenation order along the output dim that vLLM's own weight_loader uses.
PACKED_MODULES = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def _is_nvfp4_w4a4(module) -> bool:
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4LinearMethod,
    )

    return isinstance(getattr(module, "quant_method", None), ModelOptNvFp4LinearMethod)


def _assert_supported_kernel(module, name: str) -> None:
    from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
        FlashInferCutlassNvFp4LinearKernel,
    )

    kernel = module.quant_method.kernel
    if not isinstance(kernel, FlashInferCutlassNvFp4LinearKernel):
        # Every other NVFP4 kernel (trtllm shuffle, marlin repack, fbgemm)
        # applies a different post-load transform.  Reproducing the cutlass
        # one against them would write plausible-looking garbage, so refuse.
        raise RuntimeError(
            f"{name}: NVFP4 re-sync only implements the "
            f"FlashInferCutlassNvFp4LinearKernel layout, but this layer uses "
            f"{type(kernel).__name__}"
        )


def plan_resync(vllm_model) -> dict[str, tuple[str, ...]]:
    """Map each NVFP4 layer to the HF weight names that feed it.

    Returned once at startup so the per-step path does no name manipulation.
    """
    plan: dict[str, tuple[str, ...]] = {}
    for name, module in vllm_model.named_modules():
        if not _is_nvfp4_w4a4(module):
            continue
        _assert_supported_kernel(module, name)
        leaf = name.rsplit(".", 1)[-1]
        if leaf in PACKED_MODULES:
            prefix = name[: -len(leaf)]
            plan[name] = tuple(f"{prefix}{p}.weight" for p in PACKED_MODULES[leaf])
        else:
            plan[name] = (f"{name}.weight",)
    if not plan:
        raise RuntimeError(
            "no NVFP4 W4A4 linear layers found in the rollout engine; it was "
            "not built from an NVFP4 checkpoint, so the sampler would serve "
            "BF16 while the actor trains against NVFP4"
        )
    return plan


def resync_nvfp4_weights(
    vllm_model,
    weights: Mapping[str, torch.Tensor],
    plan: Mapping[str, tuple[str, ...]],
) -> set[str]:
    """Overwrite every NVFP4 layer from ``weights``; return the names consumed.

    The caller passes the remaining (unconsumed) names to ``load_weights`` as
    usual -- embeddings, norms and the LM head are not quantized.
    """
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        pad_nvfp4_weight_for_cutlass,
        swizzle_blockscale,
    )
    from vllm._custom_ops import scaled_fp4_quant

    consumed: set[str] = set()
    modules = dict(vllm_model.named_modules())

    for layer_name, sources in plan.items():
        missing = [s for s in sources if s not in weights]
        if missing:
            raise KeyError(
                f"{layer_name}: weight sync did not provide {missing}; refusing "
                "to leave a stale NVFP4 layer in the engine"
            )
        layer = modules[layer_name]

        parts = [weights[s] for s in sources]
        w = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        w = w.to(device=layer.weight.device, dtype=torch.bfloat16)

        # scaled_fp4_quant takes the *inverse* global scale: apply_weights
        # passes input_global_scale_inv, and weights follow the same
        # convention.  weight_global_scale is amax/(6*448), frozen at load.
        gs_inv = (1.0 / layer.weight_global_scale).to(torch.float32)
        packed, block_scale = scaled_fp4_quant(w, gs_inv, is_sf_swizzled_layout=False)

        # Mirror FlashInferCutlassNvFp4LinearKernel.process_weights_after_loading
        # exactly, recomputed from raw rather than mutated in place.
        new_scale = swizzle_blockscale(block_scale)
        padded, padding_cols = pad_nvfp4_weight_for_cutlass(packed)

        # copy_ rather than rebind.  The rollout engine runs with CUDA graphs
        # (enforce_eager=false, cudagraph_mode=FULL_DECODE_ONLY), and a captured
        # graph holds the *address* of layer.weight -- assigning a fresh tensor
        # to .data would leave the graph replaying against the old buffer and
        # silently serve stale weights.  Shapes are fixed for the life of the
        # engine, so an in-place copy is always valid after the first step.
        if layer.weight.data.shape != padded.shape:
            raise RuntimeError(
                f"{layer_name}: re-quantized weight {tuple(padded.shape)} does "
                f"not match the engine's {tuple(layer.weight.data.shape)}"
            )
        if layer.weight_scale.data.shape != new_scale.shape:
            raise RuntimeError(
                f"{layer_name}: re-quantized block scale {tuple(new_scale.shape)} "
                f"does not match the engine's {tuple(layer.weight_scale.data.shape)}"
            )
        layer.weight.data.copy_(padded)
        layer.weight_scale.data.copy_(new_scale.view(layer.weight_scale.data.dtype))
        layer.weights_padding_cols = padding_cols

        consumed.update(sources)

    return consumed


# Names that vLLM folds into an NVFP4 linear.  Anything matching these must be
# consumed by the re-sync; if one reaches load_weights it lands in
# default_weight_loader and asserts on a shape mismatch far from the cause
# (job 22932: "load weight [1536,1536] into parameter [2048,768]").
_NVFP4_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj")


def resync_and_load(vllm_model, named_weights: Iterable, plan) -> set[str]:
    """Full replacement for ``model.load_weights`` on an NVFP4 engine."""
    weights = dict(named_weights)
    consumed = resync_nvfp4_weights(vllm_model, weights, plan)
    rest = [(k, v) for k, v in weights.items() if k not in consumed]

    # verl runs the workers with VLLM_LOGGING_LEVEL=WARN, so info() is invisible
    # in the training logs.  These two facts are the only way to tell a working
    # re-sync from a silent fall-through, so they go out at warning level.
    LOGGER.warning(
        "NVFP4 re-sync: %d layers re-quantized, %d/%d source tensors consumed, "
        "%d passed to load_weights",
        len(plan), len(consumed), len(weights), len(rest),
    )

    # Exact suffix, not substring: ".down_proj.weight" is a prefix of
    # ".down_proj.weight_quantizer._amax", and matching loosely flagged all 196
    # modelopt buffers as leaks on j23970.
    leaked = [
        k for k, _ in rest
        if any(k.endswith(f".{leaf}.weight") for leaf in _NVFP4_LEAVES)
    ]
    if leaked:
        raise RuntimeError(
            f"NVFP4 re-sync missed {len(leaked)} quantized weights, e.g. "
            f"{sorted(leaked)[:4]}; handing them to load_weights would write "
            f"BF16 into packed uint8 parameters. Plan covers {len(plan)} layers "
            f"with sources like {sorted(next(iter(plan.values())))[:2]}."
        )

    # The FSDP actor carries modelopt's quantizer buffers (731 tensors vs the
    # 196 real weights on j23970).  vLLM has no parameters by those names, so
    # they must not reach load_weights.
    rest = [(k, v) for k, v in rest if "_quantizer." not in k]
    vllm_model.load_weights(rest)
    return consumed
