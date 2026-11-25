"""Opt-in vLLM 0.11 Qwen2 fused-QKV KV fake-quantization prototype.

This module is deliberately separate from :mod:`opqad.integrations.verl_plugin`.  Importing
it does nothing unless ``QAT_KV_VLLM_ENABLE=1`` (or the explicit install
function is called).  It is an *inference fake/dequant* path: values passed to
vLLM's cache writer are still BF16/FP16 and no integer cache is allocated.

The target is the Qwen2 implementation used by
``DeepSeek-R1-Distill-Qwen-*`` in the pinned vLLM 0.11.0 environment.  The
first prototype intentionally accepts only TP=1, ``external_launcher`` and
eager execution.  Those restrictions make failures visible instead of
silently producing a different per-shard quantization rule.

The important numerical detail is that QKV is evaluated *without* bias.  A
normal vLLM fused projection adds the learned bias in BF16; subtracting a large
K bias afterwards can destroy the small token-dependent residual.  We therefore
set ``skip_bias_add`` on each QKV layer, quantize the K/V residual, and add all
three bias slices back explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import functools
import logging
import os
import threading
from typing import Any, Optional

import torch


LOGGER = logging.getLogger(__name__)

_INIT_MARKER = "__qat_vllm_qwen2_kv_original_init_v1__"
_FORWARD_MARKER = "__qat_vllm_qwen2_kv_original_forward_v1__"
_SETTINGS_MARKER = "__qat_vllm_qwen2_kv_settings_v1__"
_INSTANCE_MARKER = "_qat_vllm_qwen2_kv_instance_v1"
# ``_qat_vllm_kv_hits`` lives on each attention instance.  Keep the model
# identity/counter evidence names centralized so the plugin and downstream
# diagnostics cannot accidentally fall back to the process-global counters.
_MODEL_COUNTER_AVAILABLE = "model_scoped_counter_available"
_MODEL_IDENTITY = "model_identity"
_MODEL_FORWARD_CALLS = "model_forward_calls"
_STATE_LOCK = threading.RLock()
_STATE: dict[str, Any] = {
    "installed": False,
    "enabled": False,
    "hits": 0,
    "bits": None,
    "layerwise": False,
    "preserve_projection_bias": True,
    "require_external_launcher": True,
    "require_tp_size": 1,
    # ``attention_instances`` counts Qwen2Attention objects created after the
    # patch is installed.  ``forwarded_instances`` counts the subset that has
    # executed at least one patched forward.  Keep ``patched_instances`` as a
    # backwards-compatible alias for the latter; ``hits``/``forward_calls``
    # count total forward invocations (not unique modules).
    "attention_instances": 0,
    "forwarded_instances": 0,
    "patched_instances": 0,
    "forward_calls": 0,
    "last_shape": None,
    "runtime_backend": None,
    "runtime_backend_evidence": None,
    "reason": "not installed",
}


@dataclass(frozen=True)
class VLLMQwen2KVSettings:
    """Validated settings for the prototype patch."""

    bits: int = 4
    layerwise: bool = False
    preserve_projection_bias: bool = True
    require_external_launcher: bool = True
    require_tp_size: int = 1
    validate_runtime: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or not isinstance(self.bits, int):
            raise TypeError("bits must be an integer")
        if self.bits < 2 or self.bits > 16:
            raise ValueError(f"prototype bits must be in [2, 16], got {self.bits}")
        if not isinstance(self.layerwise, bool):
            raise TypeError("layerwise must be a bool")
        if not isinstance(self.preserve_projection_bias, bool):
            raise TypeError("preserve_projection_bias must be a bool")
        if self.require_tp_size != 1:
            raise ValueError("the prototype is intentionally restricted to TP=1")
        if not isinstance(self.validate_runtime, bool):
            raise TypeError("validate_runtime must be a bool")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _settings_from_env() -> VLLMQwen2KVSettings:
    """Resolve explicit environment settings; malformed opt-ins fail closed."""

    bits_text = os.environ.get("QAT_KV_BITS")
    if bits_text is None:
        raise ValueError("QAT_KV_BITS is required when QAT_KV_VLLM_ENABLE=1")
    try:
        bits = int(bits_text, 10)
    except ValueError as exc:
        raise ValueError(f"QAT_KV_BITS must be an integer, got {bits_text!r}") from exc
    group_size = os.environ.get("QAT_KV_GROUP_SIZE", "-1")
    if group_size not in {"-1", "0"}:
        raise ValueError(
            "the vLLM prototype has no group-size implementation; leave "
            "QAT_KV_GROUP_SIZE unset or -1"
        )
    return VLLMQwen2KVSettings(
        bits=bits,
        layerwise=_env_bool(
            "QAT_KV_VLLM_LAYERWISE",
            _env_bool("QAT_KV_LAYERWISE", False),
        ),
        preserve_projection_bias=_env_bool(
            "QAT_KV_VLLM_PRESERVE_BIAS",
            _env_bool("QAT_KV_PRESERVE_BIAS", True),
        ),
        require_external_launcher=_env_bool("QAT_KV_VLLM_REQUIRE_EXTERNAL", True),
        require_tp_size=1,
        validate_runtime=True,
    )


def _uni_backend_evidence() -> tuple[bool, tuple[str, ...]]:
    """Return explicit evidence that a ``uni`` backend is external-launcher.

    vLLM can normalize the requested ``external_launcher`` executor to the
    ``uni``/uniprocess backend for TP=1.  A bare ``uni`` value is ambiguous:
    it is also the ordinary standalone single-process default.  The vLLM
    external-launcher path explicitly disables V1 multiprocessing, so accept
    that value only when it is visible in the environment, or when the caller
    supplies the dedicated opt-in marker.  Do not infer intent from rank
    variables alone; those are commonly present in standalone smoke shells.
    """

    evidence: list[str] = []
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "").strip() == "0":
        evidence.append("VLLM_ENABLE_V1_MULTIPROCESSING=0")
    marker = os.environ.get("QAT_KV_VLLM_EXTERNAL_LAUNCHER")
    if marker is not None and _env_bool("QAT_KV_VLLM_EXTERNAL_LAUNCHER", False):
        evidence.append("QAT_KV_VLLM_EXTERNAL_LAUNCHER=1")
    return bool(evidence), tuple(evidence)


def _llm_qat_fake_quant(values: torch.Tensor, settings: VLLMQwen2KVSettings) -> torch.Tensor:
    """Forward-equivalent rank-2 implementation of the LLM-QAT SymQuantizer.

    vLLM presents packed Q/K/V as ``[num_tokens, channels]``.  The reference
    operator uses one max-absolute scale per row when ``layerwise=False``;
    this is exactly the reduction below.  The returned values remain in the
    input dtype.  The STE wrapper is harmless in inference and keeps this
    helper usable in a diagnostic autograd call.
    """

    if values.ndim != 2:
        raise ValueError(f"fused QKV residual must be rank 2, got {tuple(values.shape)}")
    if not values.is_floating_point():
        raise TypeError(f"fused QKV residual must be floating point, got {values.dtype}")
    if values.numel() == 0:
        return values

    calc = values
    if settings.layerwise:
        max_input = torch.max(torch.abs(calc)).expand_as(calc)
    else:
        max_input = torch.max(torch.abs(calc), dim=-1, keepdim=True)[0]
        max_input = max_input.expand_as(calc).detach()
    scale = (2 ** (settings.bits - 1) - 1) / (max_input + 1e-6)
    dequant = torch.round(calc * scale).div(scale + 1e-6)
    return calc + (dequant - calc).detach()


def _runtime_check(
    settings: VLLMQwen2KVSettings,
    module: Any | None = None,
) -> None:
    """Check constraints at the first model forward, after vLLM init."""

    # Imports are intentionally lazy: the module remains importable in the HF
    # training environment and has no vLLM side effect while disabled.
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_world_size

    captured = getattr(module, "_qat_vllm_init_runtime_contract", None)
    try:
        tp_size = get_tensor_model_parallel_world_size()
    except Exception as exc:
        # A few vLLM worker contexts tear down the parallel-state accessor
        # before the first model call.  Fall back only to the constructor
        # snapshot; if it is absent, fail closed with an actionable error.
        captured_parallel = (
            captured.get("parallel_config")
            if isinstance(captured, dict)
            else None
        )
        tp_size = getattr(captured_parallel, "tensor_parallel_size", None)
        if tp_size is None and isinstance(captured_parallel, dict):
            tp_size = captured_parallel.get("tensor_parallel_size")
        if tp_size is None:
            raise RuntimeError(
                "Qwen2 fused KV prototype could not verify tensor parallel size: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    if tp_size != settings.require_tp_size:
        raise RuntimeError(
            "Qwen2 fused KV prototype requires TP=1; "
            f"vLLM reports tensor parallel size {tp_size}"
        )

    try:
        config = get_current_vllm_config()
    except Exception as exc:
        # Some vLLM worker/compiled contexts deliberately do not expose a
        # current config.  The constructor capture below is the authoritative
        # fallback; retain the exception only for the eventual fail-closed
        # diagnostic if no capture exists.
        config = None
        current_config_error = f"{type(exc).__name__}: {exc}"
    else:
        current_config_error = None
    # ``get_current_vllm_config`` is scoped to model construction/forward
    # helpers in vLLM 0.11.  In the V1 uniprocess executor it is commonly
    # cleared by the time the model's first real forward runs (and returns a
    # warning-producing default object).  Qwen2Attention.__init__ is executed
    # while the real config *is* installed, so the patched constructor stores
    # a small immutable contract on each attention module.  Prefer that
    # captured contract whenever the current object has no model_config; this
    # keeps the guard strict without requiring an unverifiable environment
    # override.
    current_model_config = getattr(config, "model_config", None) if config is not None else None
    if current_model_config is None and isinstance(captured, dict):
        config = captured

    parallel = getattr(config, "parallel_config", None) if config is not None else None
    if isinstance(config, dict):
        parallel = config.get("parallel_config")
    backend = getattr(parallel, "distributed_executor_backend", None)
    if isinstance(parallel, dict):
        backend = parallel.get("distributed_executor_backend")
    backend_name = str(backend).strip().lower() if backend is not None else None
    backend_evidence: tuple[str, ...] = ()
    if settings.require_external_launcher:
        if backend_name == "external_launcher":
            backend_evidence = ("backend=external_launcher",)
        elif backend_name == "uni":
            # vLLM 0.11 may expose the external-launcher TP=1 path as ``uni``.
            # A plain standalone ``uni`` process must remain rejected; only an
            # explicit vLLM marker (or the setting vLLM itself applies for the
            # external-launcher path) can disambiguate it.
            allowed, marker_evidence = _uni_backend_evidence()
            if not allowed:
                raise RuntimeError(
                    "Qwen2 fused KV prototype received ambiguous vLLM backend "
                    "'uni'; refusing ordinary standalone uniprocess execution. "
                    "Set VLLM_ENABLE_V1_MULTIPROCESSING=0 or explicitly set "
                    "QAT_KV_VLLM_EXTERNAL_LAUNCHER=1 when using external_launcher."
                )
            backend_evidence = ("backend=uni", *marker_evidence)
        else:
            raise RuntimeError(
                "Qwen2 fused KV prototype requires vLLM external_launcher (or "
                "an explicitly marked TP=1 uni backend); "
                f"got {backend!r}. Python monkeypatches are not reliable in the "
                "spawned async/Ray engine path."
            )
    with _STATE_LOCK:
        _STATE["runtime_backend"] = backend_name
        _STATE["runtime_backend_evidence"] = list(backend_evidence)

    model_config = getattr(config, "model_config", None) if config is not None else None
    if isinstance(config, dict):
        model_config = config.get("model_config")
    if model_config is None:
        detail = f" ({current_config_error})" if current_config_error else ""
        raise RuntimeError(
            "Qwen2 fused KV prototype could not verify enforce_eager=true: "
            "the current vLLM config has no model_config" + detail
        )
    if not bool(getattr(model_config, "enforce_eager", False)):
        raise RuntimeError(
            "Qwen2 fused KV prototype requires enforce_eager=true; "
            "the dynamic class patch is not approved for torch.compile/CUDA graphs"
        )

    cache_config = getattr(config, "cache_config", None)
    if isinstance(config, dict):
        cache_config = config.get("cache_config")
    cache_dtype = getattr(cache_config, "cache_dtype", "auto")
    if cache_dtype != "auto":
        raise RuntimeError(
            "Qwen2 fused KV prototype expects cache_dtype=auto (BF16/FP16 fake "
            f"cache); got {cache_dtype!r}, which would apply a second native cache quantizer"
        )


def _split_bias(qkv_bias: Optional[torch.Tensor], q_size: int, kv_size: int, *, device, dtype):
    if qkv_bias is None:
        return None, None, None
    if not isinstance(qkv_bias, torch.Tensor) or qkv_bias.ndim != 1:
        raise RuntimeError(
            "Qwen2 fused KV prototype expected a rank-1 qkv_proj.bias, "
            f"got {type(qkv_bias)!r} shape={getattr(qkv_bias, 'shape', None)!r}"
        )
    expected = q_size + 2 * kv_size
    if qkv_bias.numel() != expected:
        raise RuntimeError(
            "Qwen2 fused KV prototype bias width mismatch: "
            f"expected {expected}, got {qkv_bias.numel()}"
        )
    bias = qkv_bias.detach().to(device=device, dtype=dtype)
    return bias[:q_size], bias[q_size : q_size + kv_size], bias[q_size + kv_size :]


def _iter_attention_modules(model: Any):
    """Yield Qwen2-like attention modules without importing a vLLM class.

    This structural probe works through vLLM V0/V1 wrappers and is also usable
    in a tiny CPU mock.  It deliberately does not infer a quantization policy;
    :func:`validate_qwen2_model` checks the class marker and fused shape.
    """

    modules = getattr(model, "modules", None)
    if not callable(modules):
        return
    try:
        iterator = modules()
    except Exception:
        return
    for module in iterator:
        if (
            getattr(module, "qkv_proj", None) is not None
            and hasattr(module, "q_size")
            and hasattr(module, "kv_size")
        ):
            yield module


def validate_qwen2_model(model: Any, *, strict: bool = False) -> dict[str, Any]:
    """Inspect a constructed vLLM model for fused-QKV patch invariants.

    The check is side-effect free.  ``strict=True`` raises when a visible
    Qwen2-like attention is not patched, which is useful at a production
    rollout boundary; the default returns a JSON-serialisable diagnostic for
    wrappers that hide the underlying model object.
    """

    result: dict[str, Any] = {
        "model_seen": model is not None,
        # This is intentionally process-local.  It is only used to match a
        # rollout's construction-time and post-generation probes in the same
        # worker; it is not a checkpoint or cross-process identifier.
        _MODEL_IDENTITY: (
            None
            if model is None
            else f"{type(model).__module__}.{type(model).__qualname__}@{id(model):x}"
        ),
        "attention_count": 0,
        "patched_count": 0,
        "bias_count": 0,
        "skip_bias_add_count": 0,
        # These counters are scoped to the visible model rather than the
        # process-global patch state.  The latter can contain hits from an
        # earlier rollout if a worker rebuilds its engine, so strict policy
        # checks must prefer per-model evidence whenever available.
        "forwarded_count": 0,
        "forward_calls": 0,
        # Strict callers must require this marker before consuming
        # ``forward_calls``.  A visible model without per-instance counters is
        # structurally inspectable but provides no freshness evidence.
        _MODEL_COUNTER_AVAILABLE: False,
        "errors": [],
    }
    if model is None:
        result["errors"].append("model_unavailable")
        if strict:
            raise RuntimeError("cannot validate vLLM Qwen2 patch: model object is unavailable")
        result["strict_ok"] = False
        return result

    model_settings = getattr(model.__class__, _SETTINGS_MARKER, None)
    model_counter_available = True
    for index, attention in enumerate(_iter_attention_modules(model)):
        result["attention_count"] += 1
        raw_instance_hits = getattr(attention, "_qat_vllm_kv_hits", None)
        try:
            instance_hits = int(raw_instance_hits)
            if instance_hits < 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            # Do not silently turn a missing/malformed model-scoped counter
            # into zero.  In particular, the process-global ``_STATE['hits']``
            # may be non-zero from an earlier engine and is not evidence for
            # this model.
            model_counter_available = False
            instance_hits = 0
            result["errors"].append(f"attention[{index}]:model_counter_unavailable")
        if instance_hits > 0:
            result["forwarded_count"] += 1
            result["forward_calls"] += instance_hits
        module_settings = getattr(attention.__class__, _SETTINGS_MARKER, None)
        if module_settings is None:
            result["errors"].append(f"attention[{index}]:class_not_patched")
        else:
            result["patched_count"] += 1
            if getattr(attention.__class__, "forward", None) is not _patched_forward:
                result["errors"].append(f"attention[{index}]:forward_not_patched")
            if model_settings is not None and module_settings != model_settings:
                result["errors"].append(f"attention[{index}]:settings_mismatch")

        try:
            expected = int(attention.q_size) + 2 * int(attention.kv_size)
        except (TypeError, ValueError, AttributeError):
            expected = None
            result["errors"].append(f"attention[{index}]:invalid_qkv_sizes")

        qkv_proj = getattr(attention, "qkv_proj", None)
        if bool(getattr(qkv_proj, "skip_bias_add", False)):
            result["skip_bias_add_count"] += 1
        else:
            result["errors"].append(f"attention[{index}]:skip_bias_add_false")

        bias = getattr(qkv_proj, "bias", None)
        if bias is not None:
            result["bias_count"] += 1
            try:
                bias_width = int(bias.numel())
            except (TypeError, ValueError, AttributeError):
                bias_width = -1
            if expected is not None and bias_width != expected:
                result["errors"].append(
                    f"attention[{index}]:bias_width_{bias_width}_expected_{expected}"
                )

    if result["attention_count"] == 0:
        result["errors"].append("no_qwen2_attention_modules")
        model_counter_available = False
    result[_MODEL_COUNTER_AVAILABLE] = bool(model_counter_available)
    result["settings_installed"] = model_settings is not None
    result["strict_ok"] = not result["errors"]
    if strict and result["errors"]:
        raise RuntimeError("vLLM Qwen2 patch validation failed: " + ", ".join(result["errors"]))
    return result


def model_from_rollout(rollout: Any) -> Any:
    """Best-effort extraction of the underlying vLLM model for diagnostics."""

    if rollout is None:
        return None
    engines = [getattr(rollout, "inference_engine", None), getattr(rollout, "engine", None)]
    paths = (
        ("llm_engine", "model_executor", "driver_worker", "worker", "model_runner", "model"),
        ("model_executor", "driver_worker", "worker", "model_runner", "model"),
        ("worker", "model_runner", "model"),
    )
    for engine in engines:
        if engine is None:
            continue
        for path in paths:
            current = engine
            try:
                for attr in path:
                    current = getattr(current, attr)
            except (AttributeError, TypeError):
                continue
            if current is not None:
                return current
    return None


def rollout_metadata(rollout: Any = None) -> dict[str, Any]:
    """Return patch state plus optional constructed-model validation."""

    result = patch_metadata()
    if rollout is not None:
        model = model_from_rollout(rollout)
        result["model_validation"] = validate_qwen2_model(model, strict=False)
        result["model_seen"] = model is not None
        validation = result["model_validation"]
        if isinstance(validation, dict):
            # Explicit aliases make it easy for integration shims to consume
            # model-scoped evidence without falling back to _STATE['hits'].
            result[_MODEL_IDENTITY] = validation.get(_MODEL_IDENTITY)
            result[_MODEL_COUNTER_AVAILABLE] = bool(
                validation.get(_MODEL_COUNTER_AVAILABLE, False)
            )
            if result[_MODEL_COUNTER_AVAILABLE]:
                result["model_forwarded_instances"] = int(
                    validation.get("forwarded_count", 0) or 0
                )
                result[_MODEL_FORWARD_CALLS] = int(
                    validation.get("forward_calls", 0) or 0
                )
            else:
                # Preserve an explicit unavailable marker.  Returning a
                # numeric zero here would invite consumers to mistake it for
                # a fresh no-hit model or to fall back to global counters.
                result["model_forwarded_instances"] = None
                result[_MODEL_FORWARD_CALLS] = None
    return result


def _patched_forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    settings: VLLMQwen2KVSettings = getattr(self.__class__, _SETTINGS_MARKER)
    if settings.validate_runtime and not getattr(self, "_qat_vllm_runtime_checked", False):
        _runtime_check(settings, self)
        self._qat_vllm_runtime_checked = True

    qkv_result = self.qkv_proj(hidden_states)
    if isinstance(qkv_result, tuple):
        qkv_without_bias = qkv_result[0]
        returned_bias = qkv_result[1] if len(qkv_result) > 1 else None
    else:
        qkv_without_bias = qkv_result
        returned_bias = None

    q_size = int(self.q_size)
    kv_size = int(self.kv_size)
    if qkv_without_bias.shape[-1] != q_size + 2 * kv_size:
        raise RuntimeError(
            "Qwen2 fused KV prototype qkv output width mismatch: "
            f"expected {q_size + 2 * kv_size}, got {qkv_without_bias.shape[-1]}"
        )
    q0, k0, v0 = qkv_without_bias.split([q_size, kv_size, kv_size], dim=-1)
    q_bias, k_bias, v_bias = _split_bias(
        returned_bias if returned_bias is not None else getattr(self.qkv_proj, "bias", None),
        q_size,
        kv_size,
        device=qkv_without_bias.device,
        dtype=qkv_without_bias.dtype,
    )

    if q_bias is not None:
        q = q0 + q_bias
        if settings.preserve_projection_bias:
            k = _llm_qat_fake_quant(k0, settings) + k_bias
            v = _llm_qat_fake_quant(v0, settings) + v_bias
        else:
            k = _llm_qat_fake_quant(k0 + k_bias, settings)
            v = _llm_qat_fake_quant(v0 + v_bias, settings)
    else:
        q = q0
        k = _llm_qat_fake_quant(k0, settings)
        v = _llm_qat_fake_quant(v0, settings)

    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)

    self._qat_vllm_kv_hits = int(getattr(self, "_qat_vllm_kv_hits", 0)) + 1
    self._qat_vllm_last_kv_shape = (tuple(k.shape), tuple(v.shape))
    with _STATE_LOCK:
        _STATE["hits"] = int(_STATE.get("hits", 0)) + 1
        _STATE["forward_calls"] = int(_STATE.get("forward_calls", 0)) + 1
        if not bool(getattr(self, "_qat_vllm_seen_forward", False)):
            # Mark under the same lock as the global counters so concurrent
            # inference requests cannot count one module twice.
            self._qat_vllm_seen_forward = True
            _STATE["forwarded_instances"] = int(_STATE.get("forwarded_instances", 0)) + 1
            _STATE["patched_instances"] = int(_STATE.get("forwarded_instances", 0))
        _STATE["last_shape"] = {"k": list(k.shape), "v": list(v.shape)}
    return output


def install_qwen2_patch(settings: Optional[VLLMQwen2KVSettings] = None) -> dict[str, Any]:
    """Install the class patch and return a serialisable status dictionary.

    The caller should invoke this before constructing the vLLM ``LLM`` object.
    Re-installing with identical settings is idempotent; changing settings in
    a live process raises rather than stacking wrappers.
    """

    if settings is None:
        settings = _settings_from_env()
    if not isinstance(settings, VLLMQwen2KVSettings):
        raise TypeError("settings must be VLLMQwen2KVSettings")

    from vllm.model_executor.models import qwen2

    cls = qwen2.Qwen2Attention
    existing_settings = getattr(cls, _SETTINGS_MARKER, None)
    if existing_settings is not None:
        if existing_settings != settings:
            raise RuntimeError(
                "Qwen2 fused KV prototype is already installed with different settings"
            )
        return patch_metadata()

    original_init = cls.__init__
    original_forward = cls.forward

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Capture the real vLLM contract while model construction is inside
        # ``set_current_vllm_config``.  V1 may clear that context before the
        # first execution (especially for TP=1/``uni``), so a first-forward
        # check that consults only ``get_current_vllm_config`` would reject a
        # valid external-launcher model or, worse, have to trust an env flag.
        try:
            from vllm.config import get_current_vllm_config

            current = get_current_vllm_config()
            current_model_config = getattr(current, "model_config", None)
            if current_model_config is None:
                self._qat_vllm_init_runtime_contract = None
            else:
                self._qat_vllm_init_runtime_contract = {
                    "parallel_config": getattr(current, "parallel_config", None),
                    "model_config": current_model_config,
                    "cache_config": getattr(current, "cache_config", None),
                }
        except Exception as exc:
            # Do not turn a diagnostic capture failure into an unguarded
            # execution path.  The first-forward validator will fail closed
            # when neither the live nor captured config is available.
            self._qat_vllm_init_runtime_contract = {
                "capture_error": f"{type(exc).__name__}: {exc}"
            }
        qkv_proj = getattr(self, "qkv_proj", None)
        if qkv_proj is None:
            raise RuntimeError("Qwen2Attention has no qkv_proj after initialization")
        # Keep the projection residual free of bias.  QKVParallelLinear.forward
        # then returns (output_without_bias, self.bias) without changing the
        # packed local layout.
        qkv_proj.skip_bias_add = True
        setattr(self, _INSTANCE_MARKER, True)
        self._qat_vllm_runtime_checked = False
        self._qat_vllm_kv_hits = 0
        self._qat_vllm_seen_forward = False
        with _STATE_LOCK:
            _STATE["attention_instances"] = int(_STATE.get("attention_instances", 0)) + 1

    setattr(cls, _INIT_MARKER, original_init)
    setattr(cls, _FORWARD_MARKER, original_forward)
    setattr(cls, _SETTINGS_MARKER, settings)
    cls.__init__ = patched_init
    cls.forward = _patched_forward

    with _STATE_LOCK:
        _STATE.update(
            {
                "installed": True,
                "enabled": True,
                "hits": 0,
                "bits": settings.bits,
                "layerwise": settings.layerwise,
                "preserve_projection_bias": settings.preserve_projection_bias,
                "require_external_launcher": settings.require_external_launcher,
                "require_tp_size": settings.require_tp_size,
                "attention_instances": 0,
                "forwarded_instances": 0,
                "patched_instances": 0,
                "forward_calls": 0,
                "last_shape": None,
                "runtime_backend": None,
                "runtime_backend_evidence": None,
                "reason": "installed; waiting for first Qwen2Attention forward",
            }
        )
    LOGGER.warning(
        "QAT_VLLM_QWEN2_KV_PATCH enabled: bits=%d preserve_bias=%s TP=%d "
        "external_launcher=%s (fake/dequant cache; no packed int4 storage)",
        settings.bits,
        settings.preserve_projection_bias,
        settings.require_tp_size,
        settings.require_external_launcher,
    )
    return patch_metadata()


def install_from_environment() -> dict[str, Any]:
    """Install only when the explicit opt-in environment flag is true."""

    if not _env_bool("QAT_KV_VLLM_ENABLE", False):
        with _STATE_LOCK:
            _STATE.update({"enabled": False, "reason": "QAT_KV_VLLM_ENABLE is not set"})
        return patch_metadata()
    return install_qwen2_patch(_settings_from_env())


def install_for_external_launcher() -> dict[str, Any]:
    """Install at the VeRL ``_build_rollout`` boundary.

    Call this immediately before constructing ``vllm.LLM``.  It is a named
    wrapper so a launcher/plugin can make the ordering explicit and unit-test
    it without importing VeRL.  Disabled runs remain a no-op; malformed
    enabled settings raise instead of silently falling back to FP.
    """

    return install_from_environment()


def remove_qwen2_patch() -> bool:
    """Restore the original class methods, mainly for an isolated smoke test."""

    try:
        from vllm.model_executor.models import qwen2
    except ImportError:
        return False
    cls = qwen2.Qwen2Attention
    original_init = getattr(cls, _INIT_MARKER, None)
    original_forward = getattr(cls, _FORWARD_MARKER, None)
    if original_init is None or original_forward is None:
        return False
    cls.__init__ = original_init
    cls.forward = original_forward
    for marker in (_INIT_MARKER, _FORWARD_MARKER, _SETTINGS_MARKER):
        if hasattr(cls, marker):
            delattr(cls, marker)
    with _STATE_LOCK:
        _STATE.update(
            {
                "installed": False,
                "enabled": False,
                "hits": 0,
                "attention_instances": 0,
                "forwarded_instances": 0,
                "patched_instances": 0,
                "forward_calls": 0,
                "last_shape": None,
                "runtime_backend": None,
                "runtime_backend_evidence": None,
                "reason": "removed",
            }
        )
    return True


def patch_metadata() -> dict[str, Any]:
    """Return a copy suitable for logs/JSON metadata."""

    with _STATE_LOCK:
        return dict(_STATE)


def instance_metadata(module: Any) -> dict[str, Any]:
    """Return hit/shape diagnostics for one patched Qwen2Attention instance."""

    settings = getattr(module.__class__, _SETTINGS_MARKER, None)
    result: dict[str, Any] = {
        "patched_instance": bool(getattr(module, _INSTANCE_MARKER, False)),
        "hits": int(getattr(module, "_qat_vllm_kv_hits", 0)),
        "runtime_checked": bool(getattr(module, "_qat_vllm_runtime_checked", False)),
    }
    if settings is not None:
        result.update(asdict(settings))
    return result


# Importing this file from a VeRL external_lib is safe by default.  An explicit
# opt-in can make the import itself install the patch before vLLM constructs its
# model; all malformed settings intentionally raise instead of degrading to FP.
if _env_bool("QAT_KV_VLLM_ENABLE", False):
    install_from_environment()


__all__ = [
    "VLLMQwen2KVSettings",
    "install_qwen2_patch",
    "install_from_environment",
    "install_for_external_launcher",
    "model_from_rollout",
    "rollout_metadata",
    "validate_qwen2_model",
    "remove_qwen2_patch",
    "patch_metadata",
    "instance_metadata",
]
