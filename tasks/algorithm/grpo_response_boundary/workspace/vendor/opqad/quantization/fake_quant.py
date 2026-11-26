"""Generic KV activation fake quantization for decoder attention modules.

The original LLM-QAT code applies ``SymQuantizer`` to the output of the key
and value projections, before the projections are reshaped and rotary
embeddings are applied.  Qwen2 (including
``DeepSeek-R1-Distill-Qwen-*`` checkpoints) has separate ``k_proj`` and
``v_proj`` modules, so a forward hook on those modules is a small and safe way
to reproduce that placement without modifying a model implementation.

This module intentionally does not implement an integer cache.  The hook
returns dequantized floating-point values, just like the reference QAT path.
For a packed inference cache a separate cache implementation is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CLIP_RANGE: Tuple[float, float] = (-2.0, 2.0)
DEFAULT_PROJECTION_NAMES: Tuple[str, ...] = ("k_proj", "v_proj")


def _llm_qat_sym_quantizer():
    """Load the repository's reference operator lazily.

    Importing lazily keeps ``opqad`` importable in a minimal environment and
    avoids importing the large model module merely to inspect the helpers.
    ``opqad.quantization.uniform`` is the unmodified LLM-QAT operator (apart from a
    device-safe clip tensor conversion in this working tree).
    """

    try:
        from opqad.quantization.uniform import SymQuantizer
    except ImportError as exc:  # pragma: no cover - useful for standalone use
        raise ImportError(
            "OPQAD requires the repository's opqad.quantization.uniform.SymQuantizer; "
            "install the OPQAD package with pip install -e ."
        ) from exc
    return SymQuantizer


@dataclass(frozen=True)
class KVFakeQuantConfig:
    """Configuration for LLM-QAT-style KV fake quantization.

    ``bits`` is the activation bit width.  ``bits >= 32`` disables the
    operation.  ``layerwise=False`` is the LLM-QAT default and computes one
    symmetric scale per token (the last dimension of a ``[B, T, C]``
    projection output).  ``clip_range`` affects only the straight-through
    backward mask; the reference operator intentionally does not clamp the
    forward values.

    ``preserve_projection_bias`` is an explicit Qwen2/DeepSeek adaptation.
    Those checkpoints retain learned biases on ``k_proj``/``v_proj`` (some
    key biases are very large).  Quantizing the bias together with the small
    token-dependent residual makes the per-token scale unusable and can
    collapse generation at 4 bits.  When enabled, a projection hook
    quantizes ``projection_output - bias`` and adds the bias back in the
    projection output dtype (while keeping its gradient connected).  Because
    this hook runs after the framework's bias add, it is only an approximate
    bias-free seam for BF16/FP16 projections; an exact seam must intercept the
    affine operation before bias addition.  ``KVFakeQuantHook`` therefore has
    a separate, explicit ``exact_bias_free_recompute`` opt-in that recomputes
    a plain ``nn.Linear`` residual from the hook input with
    ``F.linear(weight, bias=None)`` before quantization.  The default remains
    ``False`` so the plain LLM-QAT operator is unchanged; biased Qwen2
    experiments should set the desired mode explicitly and record it.
    """

    bits: int = 32
    layerwise: bool = False
    preserve_projection_bias: bool = False
    clip_range: Tuple[float, float] = DEFAULT_CLIP_RANGE

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or not isinstance(self.bits, int):
            raise TypeError(f"bits must be an integer, got {type(self.bits)!r}")
        if self.bits < 1 or self.bits > 32:
            raise ValueError(f"bits must be in [1, 32], got {self.bits}")
        if not isinstance(self.layerwise, bool):
            raise TypeError(
                f"layerwise must be a bool, got {type(self.layerwise)!r}"
            )
        if not isinstance(self.preserve_projection_bias, bool):
            raise TypeError(
                "preserve_projection_bias must be a bool, got "
                f"{type(self.preserve_projection_bias)!r}"
            )
        try:
            normalized = tuple(float(x) for x in self.clip_range)
        except (TypeError, ValueError) as exc:
            raise TypeError("clip_range must contain numeric values") from exc
        if len(normalized) != 2:
            raise ValueError("clip_range must contain (minimum, maximum)")
        lo, hi = normalized
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"clip_range must satisfy finite min < max, got {normalized}")
        # Accept lists/generators at the public boundary but keep the frozen
        # config genuinely immutable and hashable.
        object.__setattr__(self, "clip_range", normalized)

    @property
    def enabled(self) -> bool:
        return self.bits < 32


def _clip_tensor(
    clip_range: Sequence[float], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create a clip tensor on the activation's device and dtype."""

    # A fresh tensor is intentional: SymQuantizer saves it for backward and
    # callers should be free to reuse a config across devices.
    return torch.tensor(tuple(float(x) for x in clip_range), device=device, dtype=dtype)


def symmetric_fake_quant(
    tensor: torch.Tensor,
    bits: int,
    *,
    layerwise: bool = False,
    clip_range: Sequence[float] = DEFAULT_CLIP_RANGE,
) -> torch.Tensor:
    """Apply the LLM-QAT symmetric fake quantizer to ``tensor``.

    For a rank-3 projection output this computes a scale independently for
    every ``[batch, token]`` row, then rounds and immediately dequantizes.  It
    preserves the reference straight-through estimator, including its
    gradient mask outside ``clip_range``.  No integer storage or packing is
    performed.

    ``bits == 32`` is an exact identity (including object identity), which is
    useful when toggling a hook from a config file.
    """

    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError(f"bits must be an integer, got {type(bits)!r}")
    if bits < 1 or bits > 32:
        raise ValueError(f"bits must be in [1, 32], got {bits}")
    if not isinstance(layerwise, bool):
        raise TypeError(f"layerwise must be a bool, got {type(layerwise)!r}")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor)!r}")
    # Keep the disabled path a true identity.  In particular, callers can
    # probe a configuration with an integer/complex placeholder tensor without
    # invoking the floating-point-only quantizer or validating unused clip
    # metadata.
    if bits >= 32:
        return tensor
    if not tensor.is_floating_point() and not tensor.is_complex():
        raise TypeError(f"KV activations must be floating point, got {tensor.dtype}")
    try:
        normalized_clip = tuple(float(x) for x in clip_range)
    except (TypeError, ValueError) as exc:
        raise TypeError("clip_range must contain numeric values") from exc
    if len(normalized_clip) != 2:
        raise ValueError("clip_range must contain (minimum, maximum)")
    if not (
        math.isfinite(normalized_clip[0])
        and math.isfinite(normalized_clip[1])
        and normalized_clip[0] < normalized_clip[1]
    ):
        raise ValueError(f"clip_range must satisfy finite min < max, got {normalized_clip}")

    # Empty tensors would make torch.max fail in the reference implementation,
    # but are harmless in a model's padded/zero-length edge path.
    if tensor.numel() == 0:
        return tensor
    if tensor.is_complex():
        raise TypeError("complex KV activations are not supported by SymQuantizer")

    clip = _clip_tensor(normalized_clip, device=tensor.device, dtype=tensor.dtype)
    return _llm_qat_sym_quantizer().apply(tensor, clip, bits, bool(layerwise))


class KVFakeQuantizer(nn.Module):
    """Reusable quantizer for already-computed key and value tensors.

    This class is useful inside a custom attention implementation where a
    module hook is inconvenient.  It returns ``(quantized_key,
    quantized_value)`` and keeps the tensors dequantized/floating point.
    """

    def __init__(
        self,
        bits: int = 32,
        *,
        layerwise: bool = False,
        clip_range: Sequence[float] = DEFAULT_CLIP_RANGE,
    ) -> None:
        super().__init__()
        self.config = KVFakeQuantConfig(
            bits=bits, layerwise=layerwise, clip_range=tuple(clip_range)
        )
        # Keep the range as a buffer so state/device moves are unsurprising,
        # while excluding it from checkpoints (it is configuration, not a
        # learned parameter).
        self.register_buffer(
            "clip_range",
            torch.tensor(self.config.clip_range, dtype=torch.float32),
            persistent=False,
        )

    @property
    def bits(self) -> int:
        return self.config.bits

    def forward(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(key_states, torch.Tensor) or not isinstance(
            value_states, torch.Tensor
        ):
            raise TypeError("key_states and value_states must be torch.Tensor objects")
        # Quantize independently: LLM-QAT computes a scale from each
        # activation row, not one shared K/V scale.
        return (
            symmetric_fake_quant(
                key_states,
                self.config.bits,
                layerwise=self.config.layerwise,
                clip_range=self.config.clip_range,
            ),
            symmetric_fake_quant(
                value_states,
                self.config.bits,
                layerwise=self.config.layerwise,
                clip_range=self.config.clip_range,
            ),
        )


def _leaf_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


class KVFakeQuantHook:
    """Install/remove projection hooks on Qwen2-style attention modules.

    By default only modules whose leaf names are ``k_proj`` or ``v_proj`` are
    selected.  This matches Qwen2 and the
    ``DeepSeek-R1-Distill-Qwen`` checkpoints (which are Qwen2 architecture,
    not DeepSeek MLA).  Actual DeepSeek-V2/V3 MLA models use fused
    ``kv_a_proj_with_mqa``/``kv_b_proj`` paths and must provide an explicit
    projection selection plus a model-specific slice/placement policy; this
    hook refuses to guess for those models.  Likewise, vLLM tensor-parallel
    implementations commonly fuse Q/K/V into ``qkv_proj`` and do not execute
    the Hugging Face projection modules, so this hook is for the HF training /
    replay path rather than a vLLM rollout worker.  If a tensor-parallel
    wrapper splits a projection's output before this hook runs, the scale is
    necessarily local to that shard; gather K/V channels first when exact
    global per-token LLM-QAT scales are required.

    ``exact_bias_free_recompute=True`` is intentionally narrower than the
    default hook: every selected target must be the builtin ``nn.Linear``
    class (not a subclass or adapter) and must carry a bias.  The hook uses
    the module's positional input to recompute the bias-free affine result,
    applies the same ``SymQuantizer``, and adds the connected bias back.  This
    extra matmul is the cost of avoiding post-forward BF16/FP16 cancellation;
    callers must opt in and should retain the emitted mode metadata.
    """

    def __init__(
        self,
        bits: int = 32,
        *,
        layerwise: bool = False,
        preserve_projection_bias: bool = False,
        exact_bias_free_recompute: bool = False,
        clip_range: Sequence[float] = DEFAULT_CLIP_RANGE,
        projection_names: Iterable[str] = DEFAULT_PROJECTION_NAMES,
    ) -> None:
        if not isinstance(preserve_projection_bias, bool):
            raise TypeError(
                "preserve_projection_bias must be a bool, got "
                f"{type(preserve_projection_bias)!r}"
            )
        if not isinstance(exact_bias_free_recompute, bool):
            raise TypeError(
                "exact_bias_free_recompute must be a bool, got "
                f"{type(exact_bias_free_recompute)!r}"
            )
        if exact_bias_free_recompute and not preserve_projection_bias:
            raise ValueError(
                "exact_bias_free_recompute requires preserve_projection_bias=True"
            )
        self.quantizer = KVFakeQuantizer(
            bits=bits, layerwise=layerwise, clip_range=tuple(clip_range)
        )
        self.preserve_projection_bias = preserve_projection_bias
        # The exact path is deliberately a hook-level opt-in.  It recomputes
        # a *plain* ``nn.Linear`` affine operation without bias from the hook
        # input, thereby avoiding cancellation that already happened in a
        # BF16/FP16 module output.  The default remains the cheaper,
        # post-forward subtraction path above.
        self.exact_bias_free_recompute = exact_bias_free_recompute
        if isinstance(projection_names, str):
            projection_names = (projection_names,)
        names = tuple(str(name) for name in projection_names)
        if not names:
            raise ValueError("projection_names must not be empty")
        if any(not name or "." in name for name in names):
            raise ValueError(
                "projection_names must contain non-empty leaf module names (without '.')"
            )
        self.projection_names = names
        self._handles: list[Any] = []
        self._hooked_names: list[str] = []
        self._install_called = False

    @property
    def bits(self) -> int:
        return self.quantizer.bits

    @property
    def hooked_names(self) -> Tuple[str, ...]:
        """Names of projection modules currently carrying a hook."""

        return tuple(self._hooked_names)

    @property
    def installed(self) -> bool:
        """Whether ``install`` has been called and not subsequently removed.

        A 32-bit configuration is an intentional no-op, so it is considered
        installed even though it owns no PyTorch hook handles.
        """

        return self._install_called

    @property
    def active(self) -> bool:
        """Whether at least one actual forward hook is registered."""

        return bool(self._handles)

    @property
    def bias_preservation_mode(self) -> str:
        """Describe the selected HF bias handling for run metadata."""

        if not self.preserve_projection_bias:
            return "disabled"
        if self.exact_bias_free_recompute:
            return "exact_bias_free_recompute"
        return "post_forward_approximate"

    def _exact_bias_free_hook_output(
        self,
        module_name: str,
        module: nn.Module,
        inputs: Tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute a plain ``nn.Linear`` output before its bias is added.

        A regular forward hook runs after ``nn.Linear.forward``.  Reusing the
        returned output would therefore require subtracting a bias from a
        potentially BF16/FP16-rounded value.  For this explicit path we use
        the original positional input and module parameters to perform the
        bias-free affine operation again, quantize that residual, and restore
        the connected bias.  Restricting the path to the exact builtin class
        is intentional: custom subclasses may override ``forward`` (or apply
        an adapter/quantizer), in which case a second ``F.linear`` would make
        an unverified semantic claim.
        """

        if module.__class__ is not nn.Linear:
            raise TypeError(
                f"exact bias-free KV recompute only supports plain nn.Linear; "
                f"projection {module_name!r} is {module.__class__.__qualname__}"
            )
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise TypeError(
                f"exact bias-free KV recompute expected one Tensor input for "
                f"projection {module_name!r}; got {len(inputs)} positional inputs"
            )
        projection_input = inputs[0]
        bias = module.bias
        if bias is None:
            # The exact mode is a bias-preservation mode.  A bias-free module
            # has no cancellation problem and should use the ordinary hook;
            # silently treating it as an exact contract would hide a model
            # mismatch in metadata.
            raise ValueError(
                f"exact bias-free KV recompute requires a bias on projection "
                f"{module_name!r}"
            )
        if projection_input.ndim < 1:
            raise ValueError(
                f"exact bias-free KV recompute input for {module_name!r} must "
                f"have at least one dimension, got {tuple(projection_input.shape)}"
            )

        # ``F.linear`` supports the same arbitrary leading dimensions as
        # ``nn.Linear``.  Run it under the caller's current autocast context so
        # the recomputed residual follows the model's actual arithmetic dtype.
        residual = F.linear(projection_input, module.weight, None)
        if tuple(residual.shape) != tuple(output.shape):
            raise RuntimeError(
                f"exact bias-free KV recompute shape mismatch for {module_name!r}: "
                f"recomputed {tuple(residual.shape)} vs module output {tuple(output.shape)}"
            )
        bias_value = bias.to(device=residual.device, dtype=residual.dtype)
        quantized = symmetric_fake_quant(
            residual,
            self.quantizer.config.bits,
            layerwise=self.quantizer.config.layerwise,
            clip_range=self.quantizer.config.clip_range,
        )
        return quantized + bias_value

    def _make_hook(self, module_name: str, module: nn.Module):
        def _hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"KV projection {module_name!r} returned {type(output)!r}; "
                    "the generic hook expects a tensor"
                )
            if self.exact_bias_free_recompute:
                return self._exact_bias_free_hook_output(
                    module_name, module, _inputs, output
                )
            # Qwen2/DeepSeek use biased K/V projections.  Preserve the learned
            # constant when requested, while retaining the identity STE from
            # the projection output to its parameters.  Keep the bias cast
            # connected to autograd: ``output`` already contains the same
            # bias, so the subtraction cancels its residual-path derivative
            # and the final addition supplies an un-gated bias gradient.  A
            # detached bias here would accidentally apply SymQuantizer's
            # [-2, 2] backward gate to the learned bias whenever the residual
            # is an outlier (and can silently freeze large-bias projections).
            bias = getattr(module, "bias", None)
            if self.preserve_projection_bias and bias is not None:
                if not isinstance(bias, torch.Tensor) or bias.ndim != 1:
                    raise TypeError(
                        f"KV projection {module_name!r} has unsupported bias "
                        f"shape/type {getattr(bias, 'shape', None)!r}"
                    )
                if output.shape[-1] != bias.shape[0]:
                    raise ValueError(
                        f"KV projection {module_name!r} output width "
                        f"{output.shape[-1]} does not match bias width {bias.shape[0]}"
                    )
                bias_value = bias.to(device=output.device, dtype=output.dtype)
                residual = output - bias_value
                quantized = symmetric_fake_quant(
                    residual,
                    self.quantizer.config.bits,
                    layerwise=self.quantizer.config.layerwise,
                    clip_range=self.quantizer.config.clip_range,
                )
                return quantized + bias_value
            return symmetric_fake_quant(
                output,
                self.quantizer.config.bits,
                layerwise=self.quantizer.config.layerwise,
                clip_range=self.quantizer.config.clip_range,
            )

        return _hook

    def install(self, model: nn.Module) -> "KVFakeQuantHook":
        """Register hooks and return ``self``.

        Calling ``install`` twice without ``remove`` raises to prevent
        accidental double quantization.  A disabled (32-bit) hook is a valid
        no-op and simply reports no hooked modules.
        """

        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model)!r}")
        if self._install_called:
            raise RuntimeError("this KVFakeQuantHook is already installed")

        if not self.quantizer.config.enabled:
            self._install_called = True
            return self

        matched = []
        for name, module in model.named_modules():
            if name and _leaf_name(name) in self.projection_names:
                matched.append((name, module))
        if not matched:
            names = ", ".join(self.projection_names)
            raise ValueError(
                f"no KV projection modules found (expected leaf names: {names}); "
                "fused DeepSeek MLA and vLLM qkv_proj paths require a "
                "model-specific hook"
            )

        if self.exact_bias_free_recompute:
            # Validate every target before registering any handle.  This keeps
            # a partially installed exact policy from surviving a later
            # custom/subclass mismatch.
            for name, module in matched:
                if module.__class__ is not nn.Linear:
                    raise TypeError(
                        "exact bias-free KV recompute only supports plain "
                        f"nn.Linear modules; {name!r} is "
                        f"{module.__class__.__qualname__}"
                    )
                if module.bias is None:
                    raise ValueError(
                        "exact bias-free KV recompute requires a bias on every "
                        f"target projection; {name!r} has bias=None"
                    )

        handles = []
        try:
            for name, module in matched:
                handles.append(module.register_forward_hook(self._make_hook(name, module)))
        except Exception:
            for handle in handles:
                handle.remove()
            raise
        self._handles = handles
        self._hooked_names = [name for name, _module in matched]
        self._install_called = True
        return self

    def remove(self) -> None:
        """Remove all hooks; safe to call repeatedly."""

        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hooked_names.clear()
        self._install_called = False

    def __enter__(self) -> "KVFakeQuantHook":
        if not self._install_called:
            raise RuntimeError("call install(model) before entering the context")
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.remove()


def attach_kv_fake_quant_hooks(
    model: nn.Module,
    bits: int = 32,
    *,
    layerwise: bool = False,
    preserve_projection_bias: bool = False,
    exact_bias_free_recompute: bool = False,
    clip_range: Sequence[float] = DEFAULT_CLIP_RANGE,
    projection_names: Iterable[str] = DEFAULT_PROJECTION_NAMES,
) -> KVFakeQuantHook:
    """Construct and install a :class:`KVFakeQuantHook` in one call."""

    hook = KVFakeQuantHook(
        bits=bits,
        layerwise=layerwise,
        preserve_projection_bias=preserve_projection_bias,
        exact_bias_free_recompute=exact_bias_free_recompute,
        clip_range=clip_range,
        projection_names=projection_names,
    )
    hook.install(model)
    return hook


__all__ = [
    "DEFAULT_CLIP_RANGE",
    "DEFAULT_PROJECTION_NAMES",
    "KVFakeQuantConfig",
    "KVFakeQuantHook",
    "KVFakeQuantizer",
    "attach_kv_fake_quant_hooks",
    "symmetric_fake_quant",
]
