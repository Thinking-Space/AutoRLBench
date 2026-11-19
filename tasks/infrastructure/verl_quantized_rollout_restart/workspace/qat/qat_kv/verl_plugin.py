"""Small, auditable VeRL external-module hook for actor-side KV fake quant.

This module is intentionally an *integration shim*, not another OPD
implementation.  It is loaded by VeRL through ``model.external_lib`` and
patches :class:`ActorRolloutRefWorker` just enough to install the existing
``qat_kv.attach_kv_fake_quant_hooks`` hook on the actor model.  The reference
and reward models are identified by the ``role`` argument and are never
modified.

The shim is deliberately conservative:

* KV bits must be explicit (``QAT_KV_BITS`` or a model/trainer config value).
  Missing bits disable the hook and publish a ``disabled_missing_bits`` state;
  malformed or conflicting values raise instead of silently falling back to
  full precision.  ``QAT_KV_REQUIRE_BITS=1`` turns the disabled state into a
  hard error, which is useful for production runs.
* The hook is the repository's LLM-QAT *fake/dequant* hook.  It does not pack
  an integer cache and cannot affect a separate vLLM engine.
* For the biased Qwen2 checkpoints used here, ``QAT_KV_PRESERVE_BIAS=1`` is
  enabled by the launcher.  It quantizes the projection residual and restores
  the learned K/V bias in the projection output dtype; without this explicit
  adaptation, the large early-layer key bias dominates a 4-bit per-token
  scale and the base model collapses before training.  The post-forward HF
  subtraction is approximate under BF16/FP16 and is recorded as such in
  metadata.
* Metadata is JSON-serialisable and explicitly says that this is not exact
  OPD.  vLLM/SGlang/TRTLLM rollouts receive a warning and
  ``rollout_kv_quantized=False``.
* ``QAT_KV_REQUIRE_QUANTIZED_ROLLOUT=1`` upgrades that advisory to a runtime
  gate: a vLLM rollout must expose a validated Qwen2 model and record at least
  one patched forward before its first generated batch is returned.  The
  structural check happens at construction and the hit check is deferred to
  that first generation (there cannot be a forward hit before generation).
* On the legacy Open-MOPD FSDP reward worker, ``_build_model`` may hardcode
  flash-attention.  The optional reward patch below temporarily selects eager
  attention when flash-attn is unavailable; it never installs KV hooks on the
  reward/teacher model.  Newer reference forks that expose an explicit
  ``attn_implementation`` setting need no special handling.

No VeRL import is required when this module is imported outside VeRL (for
example by unit tests).  ``install_worker_patches`` can be called again after
VeRL has finished importing its worker modules.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
import functools
import importlib
import importlib.util
import inspect
import logging
import math
import os
import sys
import threading
from collections.abc import Mapping
from typing import Any, Iterable, Optional, Sequence
import warnings


LOGGER = logging.getLogger(__name__)

PLUGIN_NAME = "qat_kv.verl_plugin"
PLUGIN_VERSION = "0.1"
BITS_ENV = "QAT_KV_BITS"
REQUIRE_BITS_ENV = "QAT_KV_REQUIRE_BITS"
LAYERWISE_ENV = "QAT_KV_LAYERWISE"
PRESERVE_BIAS_ENV = "QAT_KV_PRESERVE_BIAS"
# Explicit HF-only escape hatch for the pre-bias affine recompute.  The
# default remains false so existing runs retain the cheaper post-forward
# approximation.  vLLM has its own fused skip-bias seam and does not consume
# this setting.
EXACT_BIAS_FREE_ENV = "QAT_KV_EXACT_BIAS_FREE"
REWARD_ATTN_ENV = "QAT_KV_REWARD_ATTN"
VLLM_ENABLE_ENV = "QAT_KV_VLLM_ENABLE"
# Runtime policy gate.  This is intentionally separate from the launcher
# preflight flag: preflight can only describe intent, whereas this flag is
# checked after the worker has constructed/inspected its rollout engine.
REQUIRE_QUANTIZED_ROLLOUT_ENV = "QAT_KV_REQUIRE_QUANTIZED_ROLLOUT"
# The official KIVI rollout is HF-only in this integration.  It is kept as a
# separate opt-in from the older LLM-QAT projection-hook switch so a run
# cannot accidentally quantize both implementations at once.
KIVI_HF_ENABLE_ENV = "QAT_KV_KIVI_HF_ENABLE"
KIVI_GROUP_SIZE_ENV = "QAT_KV_KIVI_GROUP_SIZE"
KIVI_RESIDUAL_LENGTH_ENV = "QAT_KV_KIVI_RESIDUAL_LENGTH"

# ``apply_monkey_patch`` is called before FSDP wrapping in the pinned VeRL
# worker.  These context variables let its wrapper distinguish the actor build
# (where KIVI attention is installed) from the reward/teacher build (which must
# stay BF16 and unquantized).
_KIVI_ACTOR_BUILD_CONTEXT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "qat_kv_kivi_actor_build", default=False
)
_KIVI_CONFIG_CONTEXT: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "qat_kv_kivi_config", default=None
)
_KIVI_APPLY_PATCH_MARKER = "__qat_kv_kivi_apply_patch_v1__"

_HF_PRESERVE_BIAS_WARNING = (
    "Approximate only: HF preserve_projection_bias uses a post-forward hook; "
    "the projection output may already be BF16/FP16 when the learned bias is "
    "subtracted, so large-bias residuals can suffer cancellation.  This is "
    "not an exact bias-free affine implementation."
)
_HF_EXACT_BIAS_FREE_NOTE = (
    "Exact HF bias-free recompute is enabled only for builtin nn.Linear k/v "
    "projections; the hook recomputes F.linear(input, weight, bias=None), "
    "applies SymQuantizer, then restores the connected bias."
)

# This marker is put on classes rather than in a process-global set.  Ray can
# import an external module more than once in a worker process, and class
# markers make repeated imports/reloads harmless.
_PATCH_MARKER = "__qat_kv_verl_plugin_v1__"
_REWARD_PATCH_MARKER = "__qat_kv_reward_eager_plugin_v1__"
_MISSING = object()
_REWARD_PATCH_LOCK = threading.RLock()

# Runtime vLLM evidence is scoped to one rollout/model probe.  The counters
# exposed by ``qat_kv.vllm_qwen2_patch`` are cumulative on attention instances
# and the module-level diagnostic state is process-global; these worker-local
# fields let strict mode compare a construction-time baseline with the first
# generation probe instead of accepting an old hit.
_ROLLOUT_PROBE_COUNTER_ATTR = "_qat_kv_rollout_probe_counter"
_ROLLOUT_PROBE_PENDING_ATTR = "_qat_kv_rollout_probe_pending"
_ROLLOUT_PROBE_ID_ATTR = "_qat_kv_rollout_probe_id"
_MODEL_IDENTITY_KEY = "model_identity"
_MODEL_COUNTER_AVAILABLE_KEY = "model_scoped_counter_available"
_MODEL_FORWARD_CALLS_KEY = "model_forward_calls"


@dataclass(frozen=True)
class KVPluginSettings:
    """Resolved, validated settings used by the worker patch.

    ``bits=None`` is an intentional fail-closed/disabled state.  It is not
    equivalent to a request for an unquantized baseline: callers can inspect
    the metadata to distinguish an omitted setting from ``bits=32``.
    """

    bits: Optional[int]
    layerwise: bool = False
    preserve_projection_bias: bool = False
    clip_range: tuple[float, float] = (-2.0, 2.0)
    projection_names: tuple[str, ...] = ("k_proj", "v_proj")
    source: str = "missing"
    require_bits: bool = False
    # Keep this newly added option last among the positional fields so older
    # callers that constructed KVPluginSettings(bits, layerwise, preserve,
    # clip_range, projection_names, source, require_bits) remain compatible.
    exact_bias_free_recompute: bool = False

    @property
    def enabled(self) -> bool:
        return self.bits is not None and self.bits < 32


def _read_key(container: Any, key: str) -> Any:
    """Read one key from dict/OmegaConf/namespace-like objects."""

    if container is None:
        return _MISSING
    if isinstance(container, Mapping):
        try:
            return container[key] if key in container else _MISSING
        except Exception:
            return _MISSING
    # DictConfig and several VeRL config wrappers expose ``get`` without
    # registering as Mapping.
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            value = getter(key, _MISSING)
            if value is not _MISSING:
                return value
        except Exception:
            pass
    try:
        return getattr(container, key)
    except (AttributeError, TypeError):
        return _MISSING


def _read_path(container: Any, path: Sequence[str]) -> Any:
    current = container
    for key in path:
        current = _read_key(current, key)
        if current is _MISSING:
            return _MISSING
    return current


def _kivi_hf_enabled(config: Any = None) -> bool:
    """Return whether the explicit official-KIVI HF path was requested."""

    raw = os.environ.get(KIVI_HF_ENABLE_ENV, _MISSING)
    if raw is not _MISSING:
        return _parse_bool(raw, label=KIVI_HF_ENABLE_ENV, default=False)
    if config is None:
        config = _KIVI_CONFIG_CONTEXT.get()
    value, _ = _first_config_value(
        config,
        (
            ("kivi_hf_rollout",),
            ("kivi_enabled",),
            ("quantization", "kivi_hf_rollout"),
            ("quantization", "kivi_enabled"),
            ("mopd", "kivi_hf_rollout"),
        ),
    )
    if value is _MISSING or value is None:
        return False
    return _parse_bool(value, label="kivi_hf_rollout", default=False)


def _kivi_int_env_or_config(name: str, config: Any, paths: Sequence[Sequence[str]], default: int) -> int:
    raw = os.environ.get(name, _MISSING)
    value = raw if raw is not _MISSING else _first_config_value(config, paths)[0]
    if value is _MISSING or value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _kivi_settings(config: Any = None) -> tuple[int, int, int]:
    if config is None:
        config = _KIVI_CONFIG_CONTEXT.get()
    settings = resolve_settings(config)
    bits = settings.bits
    if bits is None:
        raise RuntimeError(
            f"{KIVI_HF_ENABLE_ENV}=1 requires explicit {BITS_ENV}=2 (or actor quantization bits=2)"
        )
    if bits != 2:
        raise ValueError(f"official KIVI experiment is fixed at 2 bits, got {bits}")
    group_size = _kivi_int_env_or_config(
        KIVI_GROUP_SIZE_ENV,
        config,
        (("kivi", "group_size"), ("quantization", "kivi_group_size")),
        32,
    )
    residual_length = _kivi_int_env_or_config(
        KIVI_RESIDUAL_LENGTH_ENV,
        config,
        (("kivi", "residual_length"), ("quantization", "kivi_residual_length")),
        32,
    )
    if residual_length % group_size:
        raise ValueError("KIVI residual_length must be divisible by group_size")
    return bits, group_size, residual_length


def _config_candidates(config: Any) -> list[Any]:
    """Return likely config roots without requiring OmegaConf."""

    if config is None:
        return []
    roots = [config]
    model = _read_key(config, "model")
    if model is not _MISSING and model is not None:
        roots.append(model)
    # Some callers pass ``config.actor_rollout_ref`` rather than the complete
    # trainer config.  Include its model subsection when present.
    arr = _read_key(config, "actor_rollout_ref")
    if arr is not _MISSING and arr is not None:
        roots.append(arr)
        arr_model = _read_key(arr, "model")
        if arr_model is not _MISSING and arr_model is not None:
            roots.append(arr_model)
    return roots


def _first_config_value(config: Any, paths: Iterable[Sequence[str]]) -> tuple[Any, str]:
    for root_index, root in enumerate(_config_candidates(config)):
        for path in paths:
            value = _read_path(root, path)
            if value is not _MISSING and value is not None:
                where = "config" if root_index == 0 else "config." + ".".join(path)
                return value, where
    return _MISSING, ""


def _parse_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not a boolean")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.strip():
        # Do not accept floating-point spellings such as ``4.0``.  This keeps
        # environment/config parsing unambiguous and auditable.
        try:
            result = int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer, got {value!r}") from exc
    else:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if result < 1 or result > 32:
        raise ValueError(f"{label} must be in [1, 32], got {result}")
    return result


def _parse_bool(value: Any, *, label: str, default: bool = False) -> bool:
    if value is _MISSING or value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{label} must be a boolean, got {value!r}")


def _parse_clip_range(value: Any) -> tuple[float, float]:
    if value is _MISSING or value is None:
        return (-2.0, 2.0)
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",")]
    else:
        try:
            pieces = list(value)
        except TypeError as exc:
            raise ValueError("QAT KV clip_range must contain two numbers") from exc
    if len(pieces) != 2:
        raise ValueError("QAT KV clip_range must contain exactly two numbers")
    try:
        lo, hi = float(pieces[0]), float(pieces[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("QAT KV clip_range must contain numeric values") from exc
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError(f"QAT KV clip_range must satisfy min < max, got {(lo, hi)}")
    return (lo, hi)


def _parse_projection_names(value: Any) -> tuple[str, ...]:
    if value is _MISSING or value is None:
        return ("k_proj", "v_proj")
    if isinstance(value, str):
        names = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        try:
            names = tuple(str(item) for item in value)
        except TypeError as exc:
            raise ValueError("QAT KV projection_names must be a sequence") from exc
    if not names or any(not name or "." in name for name in names):
        raise ValueError("QAT KV projection_names must contain non-empty leaf names")
    return names


def resolve_settings(
    config: Any = None, *, environ: Optional[Mapping[str, str]] = None
) -> KVPluginSettings:
    """Resolve KV settings from explicit environment/config values.

    ``QAT_KV_BITS`` is authoritative when present.  A different config value
    is treated as a safety error rather than silently overridden.  Config keys
    accepted for bits are ``qat_kv.bits``, ``qat_kv_bits``, ``kv_bits``,
    ``quantization.kv_bits``, ``quantization.student_kv_bits_primary``, and
    ``mopd.kv_bits`` (searched on both the complete config and its ``model``
    subsection).  ``qat_kv.preserve_projection_bias`` (or its
    ``quantization``/``mopd`` aliases) controls the Qwen2 bias adaptation.
    ``QAT_KV_EXACT_BIAS_FREE=1`` (or the matching
    ``qat_kv.exact_bias_free_recompute`` config key) is a separate explicit
    opt-in for the exact HF plain-``nn.Linear`` recompute; it requires bias
    preservation to be enabled.
    """

    env = os.environ if environ is None else environ
    env_raw = env.get(BITS_ENV, _MISSING)
    cfg_raw, cfg_source = _first_config_value(
        config,
        (
            ("qat_kv", "bits"),
            ("quantization", "kv_bits"),
            ("mopd", "kv_bits"),
            ("qat_kv_bits",),
            ("kv_bits",),
            ("quantization", "student_kv_bits_primary"),
            ("student_kv_bits_primary",),
        ),
    )

    env_bits: Optional[int] = None
    cfg_bits: Optional[int] = None
    if env_raw is not _MISSING and env_raw is not None:
        env_bits = _parse_int(env_raw, label=BITS_ENV)
    if cfg_raw is not _MISSING:
        cfg_bits = _parse_int(cfg_raw, label="config KV bits")
    if env_bits is not None and cfg_bits is not None and env_bits != cfg_bits:
        raise ValueError(
            f"conflicting KV bit settings: {BITS_ENV}={env_bits} but {cfg_source}={cfg_bits}"
        )

    bits = env_bits if env_bits is not None else cfg_bits
    if env_bits is not None and cfg_bits is not None:
        source = "env+config"
    elif env_bits is not None:
        source = "env"
    elif cfg_bits is not None:
        source = cfg_source or "config"
    else:
        source = "missing"

    layerwise_raw = env.get(LAYERWISE_ENV, _MISSING)
    if layerwise_raw is _MISSING:
        layerwise_raw, _ = _first_config_value(
            config, (("qat_kv", "layerwise"), ("quantization", "kv_layerwise"), ("kv_layerwise",))
        )
    layerwise = _parse_bool(layerwise_raw, label=LAYERWISE_ENV, default=False)

    preserve_bias_raw = env.get(PRESERVE_BIAS_ENV, _MISSING)
    if preserve_bias_raw is _MISSING:
        preserve_bias_raw, _ = _first_config_value(
            config,
            (
                ("qat_kv", "preserve_projection_bias"),
                ("quantization", "preserve_projection_bias"),
                ("mopd", "preserve_projection_bias"),
                ("preserve_projection_bias",),
            ),
        )
    preserve_projection_bias = _parse_bool(
        preserve_bias_raw, label=PRESERVE_BIAS_ENV, default=False
    )

    exact_bias_raw = env.get(EXACT_BIAS_FREE_ENV, _MISSING)
    if exact_bias_raw is _MISSING:
        exact_bias_raw, _ = _first_config_value(
            config,
            (
                ("qat_kv", "exact_bias_free_recompute"),
                ("quantization", "exact_bias_free_recompute"),
                ("mopd", "exact_bias_free_recompute"),
                ("exact_bias_free_recompute",),
                # Concise config aliases are accepted for compatibility; the
                # environment spelling remains the single canonical marker.
                ("qat_kv", "exact_bias_free"),
                ("quantization", "exact_bias_free"),
                ("mopd", "exact_bias_free"),
                ("exact_bias_free",),
            ),
        )
    exact_bias_free_recompute = _parse_bool(
        exact_bias_raw, label=EXACT_BIAS_FREE_ENV, default=False
    )
    if exact_bias_free_recompute and not preserve_projection_bias:
        raise ValueError(
            f"{EXACT_BIAS_FREE_ENV}=1 requires {PRESERVE_BIAS_ENV}=1; "
            "the exact path restores a learned projection bias"
        )
    if exact_bias_free_recompute and (bits is None or bits >= 32):
        raise ValueError(
            f"{EXACT_BIAS_FREE_ENV}=1 requires an explicit sub-32-bit KV setting; "
            f"got bits={bits!r}"
        )

    clip_raw, _ = _first_config_value(
        config, (("qat_kv", "clip_range"), ("quantization", "kv_clip_range"), ("kv_clip_range",))
    )
    clip_range = _parse_clip_range(clip_raw)
    projection_raw, _ = _first_config_value(
        config,
        (("qat_kv", "projection_names"), ("quantization", "kv_projection_names")),
    )
    projection_names = _parse_projection_names(projection_raw)
    require_raw = env.get(REQUIRE_BITS_ENV, _MISSING)
    if require_raw is _MISSING:
        require_raw, _ = _first_config_value(
            config,
            (("qat_kv", "require_bits"), ("quantization", "require_kv_bits"), ("require_kv_bits",)),
        )
    require_bits = _parse_bool(require_raw, label=REQUIRE_BITS_ENV, default=False)

    settings = KVPluginSettings(
        bits=bits,
        layerwise=layerwise,
        preserve_projection_bias=preserve_projection_bias,
        exact_bias_free_recompute=exact_bias_free_recompute,
        clip_range=clip_range,
        projection_names=projection_names,
        source=source,
        require_bits=require_bits,
    )
    if settings.bits is None and settings.require_bits:
        raise RuntimeError(
            f"{BITS_ENV} (or an explicit config KV bit field) is required; "
            "refusing to construct an unquantized actor"
        )
    return settings


def resolve_kv_bits(config: Any = None, *, environ: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """Convenience wrapper returning only the validated bit width."""

    return resolve_settings(config, environ=environ).bits


def _rollout_name(config: Any) -> Optional[str]:
    value, _ = _first_config_value(
        config,
        (("rollout", "name"), ("actor_rollout_ref", "rollout", "name")),
    )
    if value is _MISSING or value is None:
        return None
    return str(value)


def _is_external_rollout(name: Optional[str]) -> bool:
    if not name:
        return False
    normalized = name.lower()
    return normalized not in {"hf", "huggingface", "transformers"}


def _is_vllm_like(name: Optional[str]) -> bool:
    if not name:
        return False
    normalized = name.lower()
    return any(token in normalized for token in ("vllm", "sglang", "trtllm", "tensorrt"))


def _base_metadata(
    settings: KVPluginSettings,
    *,
    rollout_name: Optional[str] = None,
    config: Any = None,
) -> dict[str, Any]:
    return {
        "plugin": PLUGIN_NAME,
        "plugin_version": PLUGIN_VERSION,
        # This run is the single-teacher special case of the broader MOPD
        # family.  Keep the executable/runtime name honest; the old
        # ``mopd_topk_ppo`` label was too easy to read as the multi-domain,
        # multi-teacher paper setup.
        "algorithm": "single_teacher_topk_opd",
        "exact_opd": False,
        "quantization_kind": "fake_dequant",
        "packed_kv_cache": False,
        "actor_kv_bits": settings.bits,
        "actor_kv_fake_quantized": False,
        "status": "not_attached",
        "actor_kv_status": "not_attached",
        "hook_target": None,
        "hooked_modules": [],
        "replaced_existing_hook": False,
        # Alias retained for simple downstream metric consumers.
        "actor_kv_quantized": False,
        "teacher_kv_bits": 32,
        "teacher_kv_fake_quantized": False,
        "reward_model_kv_bits": 32,
        "reward_model_kv_fake_quantized": False,
        "teacher_untouched": True,
        "reward_model_untouched": True,
        "settings_source": settings.source,
        "layerwise": settings.layerwise,
        "preserve_projection_bias": settings.preserve_projection_bias,
        "exact_bias_free_recompute": settings.exact_bias_free_recompute,
        "hf_bias_preservation_mode": (
            "exact_bias_free_recompute"
            if settings.exact_bias_free_recompute
            else "post_forward_approximate"
            if settings.preserve_projection_bias
            else "disabled"
        ),
        # The generic HF hook receives an already materialised projection
        # output.  Under autocast this is commonly BF16, so subtracting a
        # large learned bias in the hook is an approximation (see
        # ``_HF_PRESERVE_BIAS_WARNING``).  The exact opt-in recomputes only
        # builtin nn.Linear targets before the bias add.
        "hf_preserve_bias_approximate": bool(
            settings.preserve_projection_bias and not settings.exact_bias_free_recompute
        ),
        "hf_preserve_bias_exact": (
            True
            if settings.exact_bias_free_recompute
            else False
            if settings.preserve_projection_bias
            else None
        ),
        "hf_preserve_bias_warning": (
            _HF_PRESERVE_BIAS_WARNING
            if settings.preserve_projection_bias and not settings.exact_bias_free_recompute
            else None
        ),
        # Keep a descriptive alias for consumers that use the setting's full
        # name when building a run manifest.
        "preserve_projection_bias_approximate": bool(
            settings.preserve_projection_bias and not settings.exact_bias_free_recompute
        ),
        "preserve_projection_bias_exact": (
            True if settings.exact_bias_free_recompute else False if settings.preserve_projection_bias else None
        ),
        "preserve_projection_bias_mode": (
            "exact_bias_free_recompute"
            if settings.exact_bias_free_recompute
            else "post_forward_approximate"
            if settings.preserve_projection_bias
            else "disabled"
        ),
        "preserve_projection_bias_warning": (
            _HF_PRESERVE_BIAS_WARNING
            if settings.preserve_projection_bias and not settings.exact_bias_free_recompute
            else None
        ),
        "rollout_quantized_required": _quantized_rollout_required(),
        "clip_range": list(settings.clip_range),
        "projection_names": list(settings.projection_names),
        "rollout_engine": rollout_name,
        "kivi_hf_rollout": _kivi_hf_enabled(config),
        "kivi_hf_metadata": None,
        "rollout_kv_quantized": None,
        "rollout_kv_verified": False,
        "rollout_kv_status": "not_built" if rollout_name is None else "unknown",
        # Freshness evidence is populated when a vLLM rollout is probed.  A
        # ``None``/False value is intentional: it means no model-scoped
        # counter was available and must never be replaced by global hits.
        "rollout_kv_probe_id": None,
        "rollout_kv_model_identity": None,
        "rollout_kv_model_counter_available": False,
        "rollout_kv_model_forward_calls_baseline": None,
        "rollout_kv_model_forward_calls_current": None,
        "rollout_kv_model_forward_calls_delta": None,
        # This is the high-level policy/cache mismatch flag consumed by run
        # manifests.  ``None`` means rollout construction has not completed.
        "rollout_policy_mismatch": None,
        "vllm_kv_patch": None,
        "rollout_kv_strict_pending": False,
        "rollout_kv_strict_deferred": False,
        "rollout_kv_strict_generation_failed": False,
        "rollout_kv_strict_generation_verified": False,
        # Keep the generic warning populated at actor-build time as well as in
        # the dedicated fields above.  Rollout annotation may replace it with
        # an engine-specific warning, while the dedicated HF fields remain
        # stable for manifests.
        "warning": (
            _HF_PRESERVE_BIAS_WARNING
            if settings.preserve_projection_bias and not settings.exact_bias_free_recompute
            else _HF_EXACT_BIAS_FREE_NOTE
            if settings.exact_bias_free_recompute
            else None
        ),
    }


def _publish_metadata(owner: Any, metadata: Mapping[str, Any], model: Any = None) -> dict[str, Any]:
    """Publish a plain dict on worker/model without touching VeRL config schema."""

    result = dict(metadata)
    # Keep both names: the public one is convenient for logging, while the
    # private one avoids collisions with a future VeRL field.
    try:
        setattr(owner, "qat_kv_metadata", result)
        setattr(owner, "_qat_kv_metadata", result)
    except Exception:
        LOGGER.debug("could not attach KV metadata to worker", exc_info=True)
    if model is not None:
        try:
            setattr(model, "qat_kv_metadata", dict(result))
        except Exception:
            LOGGER.debug("could not attach KV metadata to actor model", exc_info=True)
    return result


def _vllm_enable_requested() -> bool:
    """Resolve the explicit vLLM KV-patch opt-in.

    The standard external engine is intentionally *not* treated as quantized.
    A caller must opt in to the model-specific fused-QKV patch.  Invalid
    values are errors rather than a silent fallback to an unquantized rollout;
    this is important because an experiment can otherwise look successful
    while optimizing a different policy than the one that generated samples.
    """

    raw = os.environ.get(VLLM_ENABLE_ENV)
    if raw is None:
        return False
    return _parse_bool(raw, label=VLLM_ENABLE_ENV, default=False)


def _quantized_rollout_required() -> bool:
    """Whether a worker must prove a quantized rollout before proceeding.

    The check is deliberately evaluated at the rollout boundary rather than
    import time.  VeRL imports ``external_lib`` in its driver as well as in
    Ray workers, and only the latter has enough information to inspect the
    constructed engine.  Invalid values raise instead of silently selecting
    the advisory/default path.
    """

    raw = os.environ.get(REQUIRE_QUANTIZED_ROLLOUT_ENV)
    if raw is None:
        return False
    return _parse_bool(raw, label=REQUIRE_QUANTIZED_ROLLOUT_ENV, default=False)


def _record_vllm_patch_failure(worker: Any, exc: BaseException, *, status: str = "install_failed") -> dict[str, Any]:
    """Publish an explicit vLLM-patch failure without touching the rollout.

    The helper is intentionally best-effort: it must never replace the
    original exception raised by the rollout builder.  Keeping the record on
    the worker lets a failed Ray actor report *why* the opt-in was rejected
    instead of looking like an ordinary vLLM startup crash.
    """

    previous = getattr(worker, "_qat_kv_vllm_patch_metadata", None)
    # The wrapper may observe the same exception after the installer already
    # classified it (for example ``settings_mismatch``).  Preserve that more
    # specific status instead of replacing it with the generic label.
    if (
        status == "install_failed"
        and isinstance(previous, Mapping)
        and previous.get("status") in {"settings_mismatch", "inactive", "install_failed"}
    ):
        status = str(previous.get("status"))
    metadata = {
        "installed": False,
        "enabled": True,
        "status": status,
        "reason": f"{type(exc).__name__}: {exc}",
    }
    try:
        setattr(worker, "_qat_kv_vllm_patch_metadata", dict(metadata))
    except Exception:
        LOGGER.debug("could not attach vLLM patch failure metadata", exc_info=True)
    # If actor metadata already exists, update its public copy immediately;
    # the rollout annotation in the finally block will preserve this record.
    try:
        current = get_qat_kv_metadata(worker)
        if current:
            current.update(
                {
                    "rollout_kv_quantized": False,
                    "rollout_kv_status": "vllm_patch_failed",
                    "rollout_policy_mismatch": True,
                    "vllm_kv_patch": dict(metadata),
                    "warning": "vLLM KV patch installation failed: " + metadata["reason"],
                }
            )
            _publish_metadata(worker, current)
    except Exception:
        LOGGER.debug("could not publish vLLM patch failure metadata", exc_info=True)
    return metadata


def _install_vllm_rollout_patch(worker: Any, rollout_name: Optional[str]) -> dict[str, Any] | None:
    """Install the fused-QKV patch before constructing a vLLM rollout.

    ``external_launcher`` in the pinned Open-MOPD fork constructs the vLLM
    object in the Ray worker process, so importing the patch immediately before
    ``_build_rollout`` is early enough for Qwen2Attention class construction.
    Other vLLM multiprocessing modes are deliberately rejected by the patch's
    runtime checks; they need a child-process/sitecustomize integration rather
    than a misleading parent-process monkeypatch.
    """

    requested = _vllm_enable_requested()
    if not requested:
        return None
    if not _is_vllm_like(rollout_name):
        raise RuntimeError(
            f"{VLLM_ENABLE_ENV}=1 requires a vLLM-like rollout, got {rollout_name!r}"
        )

    try:
        patch_module = importlib.import_module("qat_kv.vllm_qwen2_patch")
        # ``install_for_external_launcher`` is intentionally a separate name:
        # it documents that this call happens before ``vllm.LLM`` is built and
        # gives tests/custom launchers a stable integration point.  Keep the
        # older installer as a compatibility fallback for tiny downstream
        # shims that provide only ``install_from_environment``.
        installer = getattr(patch_module, "install_for_external_launcher", None)
        if not callable(installer):
            installer = getattr(patch_module, "install_from_environment", None)
        if not callable(installer):
            raise AttributeError(
                "qat_kv.vllm_qwen2_patch exposes neither "
                "install_for_external_launcher nor install_from_environment"
            )
        status = dict(installer())
    except Exception as exc:
        _record_vllm_patch_failure(worker, exc)
        raise RuntimeError(
            "vLLM KV quantization was explicitly enabled but the Qwen2 fused-QKV "
            f"patch could not be installed: {exc}"
        ) from exc

    if not bool(status.get("installed")) or not bool(status.get("enabled")):
        error = RuntimeError(
            "vLLM KV quantization was explicitly enabled but patch status is "
            f"not active: {status}"
        )
        _record_vllm_patch_failure(worker, error, status="inactive")
        raise RuntimeError(
            str(error)
        ) from error
    settings_bits = status.get("bits")
    actor_config = getattr(worker, "config", None)
    actor_settings = resolve_settings(actor_config)
    actor_bits = actor_settings.bits
    if actor_bits is not None and settings_bits is not None and int(actor_bits) != int(settings_bits):
        error = RuntimeError(
            "HF actor and vLLM KV bit settings disagree: "
            f"actor={actor_bits}, rollout={settings_bits}"
        )
        _record_vllm_patch_failure(worker, error, status="settings_mismatch")
        raise error
    # The quantizer placement and bias policy are part of the policy identity,
    # not cosmetic metadata.  Compare them whenever the patch reports the
    # fields (older test/downstream shims may not), and publish the actor side
    # values for post-run manifests.
    consistency_checks = (
        ("layerwise", actor_settings.layerwise),
        ("preserve_projection_bias", actor_settings.preserve_projection_bias),
    )
    for field, actor_value in consistency_checks:
        if field not in status or status.get(field) is None:
            continue
        if bool(status.get(field)) != bool(actor_value):
            error = RuntimeError(
                "HF actor and vLLM KV settings disagree for "
                f"{field}: actor={actor_value}, rollout={status.get(field)}"
            )
            _record_vllm_patch_failure(worker, error, status="settings_mismatch")
            raise error
        status[f"actor_{field}"] = bool(actor_value)
    try:
        setattr(worker, "_qat_kv_vllm_patch_metadata", dict(status))
    except Exception:
        LOGGER.debug("could not attach vLLM patch metadata to worker", exc_info=True)
    return status


def get_qat_kv_metadata(owner: Any) -> dict[str, Any]:
    """Return a defensive copy of plugin metadata from a worker/model."""

    value = getattr(owner, "qat_kv_metadata", None)
    if value is None:
        value = getattr(owner, "_qat_kv_metadata", None)
    if value is None:
        value = getattr(owner, "qat_kv_reward_metadata", None)
    if value is None:
        value = getattr(owner, "_qat_kv_reward_metadata", None)
    return dict(value) if isinstance(value, Mapping) else {}


def _model_from_build_result(result: Any) -> Any:
    # VeRL's FSDP worker returns (module, optimizer, scheduler, config).  A
    # small fallback for custom workers returning the module directly keeps the
    # shim easy to test and compatible with older forks.
    if isinstance(result, Mapping):
        for key in ("model", "actor_module_fsdp", "actor_module", "module"):
            value = result.get(key, _MISSING)
            if value is not _MISSING:
                return value
    if isinstance(result, (tuple, list)) and result:
        return result[0]
    return result


def _iter_model_candidates(model: Any):
    """Yield wrapper and unwrapped model candidates without cycles."""

    queue: list[tuple[str, Any]] = [("", model)]
    seen: set[int] = set()
    while queue:
        path, current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield path, current
        for attr in ("_fsdp_wrapped_module", "module", "_orig_mod", "base_model", "model"):
            try:
                child = getattr(current, attr)
            except Exception:
                continue
            if child is not None and child is not current:
                child_path = f"{path}.{attr}" if path else attr
                queue.append((child_path, child))


def _has_projection_modules(model: Any, names: Sequence[str]) -> bool:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return False
    try:
        for module_name, _module in named_modules():
            if module_name and module_name.rsplit(".", 1)[-1] in names:
                return True
    except Exception:
        return False
    return False


def _select_hook_target(model: Any, names: Sequence[str]) -> tuple[str, Any]:
    for path, candidate in _iter_model_candidates(model):
        if _has_projection_modules(candidate, names):
            return path, candidate
    raise RuntimeError(
        "KV fake-quantization was enabled but no k_proj/v_proj modules were "
        "found. Fused MLA/qkv_proj and vLLM engines require a model-specific "
        "integration; refusing to run unquantized."
    )


def _remove_actor_hook(worker: Any, hook: Any) -> None:
    """Detach a previously installed actor hook, failing closed on errors."""

    remove = getattr(hook, "remove", None)
    if not callable(remove):
        raise RuntimeError("existing actor KV hook cannot be removed safely")
    try:
        remove()
    except Exception as exc:
        raise RuntimeError("failed to remove stale actor KV hook; refusing ambiguous quantization state") from exc
    # Keep the attributes present but neutral so custom worker diagnostics can
    # distinguish a deliberate no-op from an uninitialised object.
    setattr(worker, "_qat_kv_hook", None)
    setattr(worker, "_qat_kv_hook_target", "")
    setattr(worker, "_qat_kv_actor_model_id", None)
    setattr(worker, "_qat_kv_hook_target_id", None)


def _attach_actor_model(worker: Any, model: Any) -> dict[str, Any]:
    """Attach the existing hook to one actor model and publish metadata."""

    config = getattr(worker, "config", None)
    settings = resolve_settings(config)
    metadata = _base_metadata(settings, rollout_name=_rollout_name(config), config=config)
    metadata["worker_role"] = str(getattr(worker, "role", "actor"))

    # TurboQuant is a different operator at a different point in the network:
    # it quantizes post-RoPE inside attention, whereas the projection hook
    # below runs on pre-RoPE k_proj/v_proj outputs.  Route it out before any of
    # the legacy paths so the two can never both be installed.
    from .tq_scheme import attach_turboquant_actor, is_quantized_scheme, resolve_scheme

    scheme = resolve_scheme(config)
    # ``is_quantized_scheme``, not ``is_turboquant_scheme``: int4_per_token_head
    # shares the post-RoPE hook but is not a TurboQuant preset, and the narrow
    # test would drop it through to the legacy pre-RoPE projection hook -- a
    # different quantizer at a different point, installed without a word.
    if is_quantized_scheme(scheme):
        if _kivi_hf_enabled(config):
            raise RuntimeError(
                f"KV scheme {scheme!r} and the official KIVI HF path are mutually "
                "exclusive; both would quantize K/V"
            )
        metadata.update(attach_turboquant_actor(worker, model, scheme))
        return _publish_metadata(worker, metadata, model=model)
    metadata["scheme"] = scheme or "symmetric"

    # Official KIVI replaces the attention/cache boundary before FSDP wraps
    # the model.  Do not add the older projection hook on top of it: that would
    # quantize K/V twice and would no longer be the KIVI algorithm.
    if _kivi_hf_enabled(config):
        kivi_metadata = getattr(model, "_qat_kivi_metadata", None)
        if not isinstance(kivi_metadata, Mapping):
            raise RuntimeError(
                f"{KIVI_HF_ENABLE_ENV}=1 was requested but the actor model was not "
                "installed with the official KIVI HF attention before FSDP"
            )
        metadata.update(
            {
                "status": "enabled_kivi_hf",
                "actor_kv_status": "enabled_kivi_hf",
                "actor_kv_fake_quantized": True,
                "actor_kv_quantized": True,
                "kivi_hf_rollout": True,
                "kivi_hf_metadata": dict(kivi_metadata),
                "hook_target": "qwen2.self_attn.kivi_attention",
                "hooked_modules": [
                    f"model.layers.{i}.self_attn" for i in range(int(kivi_metadata.get("attention_modules", 0)))
                ],
            }
        )
        setattr(worker, "_qat_kv_hook", None)
        setattr(worker, "_qat_kivi_model_id", id(model))
        # Keep one concise runtime proof in the worker log.  The resolved
        # launcher settings only describe the requested path; this confirms
        # that every actor process actually installed KIVI on the model it
        # will train.
        if int(getattr(worker, "rank", 0)) == 0:
            LOGGER.info("%s: actor KIVI installed: %s", PLUGIN_NAME, metadata["kivi_hf_metadata"])
        return _publish_metadata(worker, metadata, model=model)

    existing = getattr(worker, "_qat_kv_hook", None)
    existing_installed = existing is not None and bool(getattr(existing, "installed", False))

    if settings.bits is None:
        if existing_installed:
            _remove_actor_hook(worker, existing)
        metadata.update(
            {
                "status": "disabled_missing_bits",
                "actor_kv_status": "disabled_missing_bits",
                "reason": f"{BITS_ENV} and config KV bits were not set (fail-closed)",
            }
        )
        LOGGER.warning("%s: KV fake quantization disabled because bits were not explicit", PLUGIN_NAME)
        return _publish_metadata(worker, metadata, model=model)

    # ``bits=32`` is an explicit full-precision diagnostic mode.  Keep it a
    # true no-op instead of passing an arbitrary build result to the hook
    # factory (which quite reasonably requires an ``nn.Module`` even when the
    # quantizer itself is disabled).
    if not settings.enabled:
        if existing_installed:
            _remove_actor_hook(worker, existing)
        metadata.update(
            {
                "status": "disabled_32",
                "actor_kv_status": "disabled_32",
                "reason": "explicit 32-bit setting disables the fake-quant hook",
            }
        )
        return _publish_metadata(worker, metadata, model=model)

    # Avoid a second set of hooks if a custom worker calls its model builder
    # twice with the same model/settings.  If the model or settings changed,
    # detach the stale hook before installing the new one; silently retaining
    # it would leave the rebuilt actor quantized at an unknown bit width.
    if existing_installed:
        existing_config = getattr(getattr(existing, "quantizer", None), "config", None)
        same_settings = (
            getattr(existing, "bits", None) == settings.bits
            and getattr(existing_config, "layerwise", None) == settings.layerwise
            and getattr(existing, "preserve_projection_bias", False)
            == settings.preserve_projection_bias
            and getattr(existing, "exact_bias_free_recompute", False)
            == settings.exact_bias_free_recompute
            and tuple(getattr(existing_config, "clip_range", ())) == tuple(settings.clip_range)
            and tuple(getattr(existing, "projection_names", ())) == tuple(settings.projection_names)
        )
        same_model = getattr(worker, "_qat_kv_actor_model_id", None) == id(model)
        if same_model and same_settings:
            metadata.update(
                {
                    "status": "already_attached",
                    "actor_kv_status": "enabled_fake_dequant",
                    "actor_kv_fake_quantized": True,
                    "actor_kv_quantized": True,
                    "hook_target": getattr(worker, "_qat_kv_hook_target", ""),
                    "hooked_modules": list(getattr(existing, "hooked_names", ())),
                }
            )
            return _publish_metadata(worker, metadata, model=model)
        _remove_actor_hook(worker, existing)

    replaced_existing = existing_installed

    if replaced_existing:
        metadata.update(
            {
                "replaced_existing_hook": True,
            }
        )

    try:
        target_path, target = _select_hook_target(model, settings.projection_names)
        # Import lazily so importing the plugin in a driver process does not
        # pull torch/model code into environments that only inspect
        # configuration.
        from .fake_quant import attach_kv_fake_quant_hooks

        hook = attach_kv_fake_quant_hooks(
            target,
            bits=settings.bits,
            layerwise=settings.layerwise,
            preserve_projection_bias=settings.preserve_projection_bias,
            exact_bias_free_recompute=settings.exact_bias_free_recompute,
            clip_range=settings.clip_range,
            projection_names=settings.projection_names,
        )
    except Exception as exc:
        # An enabled setting must never silently degrade to FP.  The original
        # exception is chained to retain the useful model/module traceback.
        metadata.update(
            {
                "status": "attach_failed",
                "actor_kv_status": "attach_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        _publish_metadata(worker, metadata, model=model)
        raise RuntimeError("failed to install actor KV fake-quantization hook; refusing unquantized training") from exc

    setattr(worker, "_qat_kv_hook", hook)
    setattr(worker, "_qat_kv_hook_target", target_path)
    setattr(worker, "_qat_kv_actor_model_id", id(model))
    setattr(worker, "_qat_kv_hook_target_id", id(target))
    metadata.update(
        {
            "status": "enabled_fake_dequant" if settings.enabled else "disabled_32",
            "actor_kv_status": "enabled_fake_dequant" if settings.enabled else "disabled_32",
            "actor_kv_fake_quantized": bool(settings.enabled and getattr(hook, "active", True)),
            "actor_kv_quantized": bool(settings.enabled and getattr(hook, "active", True)),
            "hook_target": target_path,
            "hooked_modules": list(getattr(hook, "hooked_names", ())),
            "hf_bias_preservation_mode": getattr(
                hook, "bias_preservation_mode", metadata.get("hf_bias_preservation_mode")
            ),
        }
    )
    return _publish_metadata(worker, metadata, model=model)


def _vllm_patch_requested() -> bool:
    """Backward-compatible spelling for the explicit vLLM opt-in parser."""

    return _vllm_enable_requested()


def _install_vllm_patch_before_rollout(worker: Any) -> dict[str, Any] | None:
    """Compatibility wrapper for callers that do not pass rollout name."""

    rollout_name = _rollout_name(getattr(worker, "config", None))
    return _install_vllm_rollout_patch(worker, rollout_name)


_STRICT_ROLLOUT_REJECT_STATUSES = frozenset(
    {
        # Installer-side failures.
        "vllm_patch_failed",
        "patch_unavailable",
        "install_failed",
        "inactive",
        "settings_mismatch",
        # Post-build probe failures/uncertainty.
        "qwen2_fused_fake_dequant_validation_failed",
        "validation_failed",
        "qwen2_fused_fake_dequant_model_unavailable",
        "model_unavailable",
        "qwen2_fused_fake_dequant_pending_hit",
        "pending_hit",
        "pending",
        "no_hit",
        # No model-specific patch was active.
        "unquantized_external_engine",
        "unquantized",
        "unknown",
        "not_built",
    }
)


def _vllm_rollout_hits(metadata: Mapping[str, Any]) -> int:
    """Read a *model-scoped* vLLM forward counter.

    ``vllm_qwen2_patch.patch_metadata()['hits']`` is deliberately not a
    fallback here: it is process-global and can retain evidence from another
    engine/rollout.  A caller must provide an explicit model-scoped counter
    marker plus ``model_forward_calls``.  Returning zero for an unavailable or
    malformed probe keeps strict mode fail-closed while leaving advisory mode
    able to publish its diagnostic metadata.
    """

    _available, _identity, current = _vllm_model_probe_values(metadata)
    return current if _available and current is not None else 0


def _safe_nonnegative_int(value: Any) -> Optional[int]:
    """Parse a counter without accepting booleans/negative values."""

    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _vllm_model_probe_values(
    metadata: Mapping[str, Any],
) -> tuple[bool, Any, Optional[int]]:
    """Extract ``(counter_available, model_identity, forward_calls)``.

    The preferred source is ``vllm_kv_patch.runtime_probe``.  For compatibility
    with a small downstream shim, explicit model-scoped fields directly on the
    patch report are also accepted.  Neither branch ever consumes the global
    ``hits`` field.
    """

    report = metadata.get("vllm_kv_patch")
    if not isinstance(report, Mapping):
        return False, None, None
    runtime_probe = report.get("runtime_probe")
    source: Mapping[str, Any]
    if isinstance(runtime_probe, Mapping):
        source = runtime_probe
    else:
        source = report

    validation = source.get("model_validation")
    if isinstance(validation, Mapping):
        available = source.get(_MODEL_COUNTER_AVAILABLE_KEY, _MISSING)
        if available is _MISSING:
            available = validation.get(_MODEL_COUNTER_AVAILABLE_KEY, False)
        identity = source.get(_MODEL_IDENTITY_KEY, _MISSING)
        if identity is _MISSING:
            identity = validation.get(_MODEL_IDENTITY_KEY)
        raw_current = source.get(_MODEL_FORWARD_CALLS_KEY, _MISSING)
        if raw_current is _MISSING:
            raw_current = validation.get(_MODEL_FORWARD_CALLS_KEY, _MISSING)
    else:
        # A report without structural model validation is not sufficient
        # evidence, even if a hand-written shim includes a numeric counter.
        available = source.get(_MODEL_COUNTER_AVAILABLE_KEY, False)
        identity = source.get(_MODEL_IDENTITY_KEY)
        raw_current = source.get(_MODEL_FORWARD_CALLS_KEY, _MISSING)

    if available is not True or identity in (_MISSING, None, ""):
        return False, None if identity is _MISSING else identity, None
    current = _safe_nonnegative_int(raw_current)
    if current is None:
        return False, identity, None
    return True, identity, current


def _vllm_rollout_forward_delta(metadata: Mapping[str, Any]) -> Optional[int]:
    """Return the current-minus-baseline model counter, if recorded."""

    delta = _safe_nonnegative_int(metadata.get("rollout_kv_model_forward_calls_delta"))
    return delta


def _record_vllm_model_probe(
    metadata: dict[str, Any],
    runtime_probe: Optional[Mapping[str, Any]],
    worker: Any,
    *,
    new_probe: bool,
) -> None:
    """Record model-scoped baseline/current/delta evidence on a worker.

    ``new_probe`` is set by the rollout builder before constructing a fresh
    engine.  On the first probe we snapshot the current per-model count as the
    baseline; later probes (the deferred first-generation check) preserve that
    baseline and compute a delta.  If the model identity changes, the count is
    treated as a new scope and a zero delta is established until a forward is
    observed.  Missing/invalid model counters are explicitly unavailable and
    never replaced with the process-global patch count.
    """

    wrapped_probe: Mapping[str, Any]
    if isinstance(runtime_probe, Mapping):
        wrapped_probe = {"vllm_kv_patch": {"runtime_probe": runtime_probe}}
    else:
        wrapped_probe = {}
    available, identity, count = _vllm_model_probe_values(wrapped_probe)
    previous_identity = metadata.get("rollout_kv_model_identity")
    previous_baseline = _safe_nonnegative_int(
        metadata.get("rollout_kv_model_forward_calls_baseline")
    )

    if not available or count is None:
        metadata.update(
            {
                "rollout_kv_probe_id": getattr(worker, _ROLLOUT_PROBE_ID_ATTR, None),
                "rollout_kv_model_identity": identity,
                "rollout_kv_model_counter_available": False,
                "rollout_kv_model_forward_calls_baseline": None,
                "rollout_kv_model_forward_calls_current": None,
                "rollout_kv_model_forward_calls_delta": None,
                "rollout_kv_model_probe_fresh": False,
            }
        )
        return

    # A fresh rollout always gets a new baseline, even when vLLM reuses the
    # same model/attention objects.  This is the key stale-hit protection.
    if (
        new_probe
        or previous_baseline is None
        or previous_identity in (None, "")
        or previous_identity != identity
    ):
        baseline = count
        delta = 0
    else:
        # A backend may recreate attention objects in place and reset their
        # counters.  Re-baseline that explicit reset instead of allowing a
        # negative delta to poison every subsequent generation check.
        baseline = previous_baseline if count >= previous_baseline else count
        delta = count - baseline

    metadata.update(
        {
            "rollout_kv_probe_id": getattr(worker, _ROLLOUT_PROBE_ID_ATTR, None),
            "rollout_kv_model_identity": identity,
            "rollout_kv_model_counter_available": True,
            "rollout_kv_model_forward_calls_baseline": baseline,
            "rollout_kv_model_forward_calls_current": count,
            "rollout_kv_model_forward_calls_delta": delta,
            "rollout_kv_model_probe_fresh": bool(delta > 0),
        }
    )


def _enforce_quantized_rollout(
    metadata: dict[str, Any],
    *,
    rollout_name: Optional[str],
    allow_pending: bool = False,
) -> None:
    """Fail closed when strict runtime policy identity was requested.

    A class-level vLLM monkeypatch is not sufficient evidence: the worker must
    have inspected a visible Qwen2 model and observed at least one patched
    forward.  A rollout has not generated anything at its construction
    boundary, however, so a structurally valid Qwen2 patch naturally has a
    zero-hit *pending* state.  ``allow_pending`` is used only at that boundary;
    the first generation call performs the non-pending check.
    """

    required = _quantized_rollout_required()
    metadata["rollout_quantized_required"] = required
    if not required:
        return

    normalized_name = str(rollout_name).lower() if rollout_name is not None else ""
    status = str(metadata.get("rollout_kv_status") or "unknown")
    quantized = metadata.get("rollout_kv_quantized")
    hits = _vllm_rollout_hits(metadata)
    model_counter_available = (
        metadata.get("rollout_kv_model_counter_available") is True
    )
    model_forward_delta = _vllm_rollout_forward_delta(metadata)
    model_identity = metadata.get("rollout_kv_model_identity")

    # The only currently supported external-engine proof is the explicit
    # Qwen2 fused-QKV prototype.  It must be both active and hit at least once.
    if _is_vllm_like(rollout_name):
        accepted = (
            quantized is True
            and status == "qwen2_fused_fake_dequant_active"
            and model_counter_available
            and model_identity not in (None, "")
            # A strict rollout must demonstrate a new patched forward after
            # this rollout's construction-time baseline.  A cumulative hit
            # from a reused model is not sufficient evidence.
            and model_forward_delta is not None
            and model_forward_delta > 0
        )
        if accepted:
            metadata["rollout_kv_verified"] = True
            metadata["rollout_kv_strict_pending"] = False
            metadata["rollout_kv_strict_deferred"] = False
            return
        runtime_probe = metadata.get("vllm_kv_patch")
        validation = (
            runtime_probe.get("runtime_probe", {}).get("model_validation")
            if isinstance(runtime_probe, Mapping)
            and isinstance(runtime_probe.get("runtime_probe"), Mapping)
            else None
        )
        # Do not treat a partial/hand-written probe as structural evidence.
        # The real validator always emits ``model_seen`` and an explicit
        # ``errors`` list (plus ``strict_ok``); requiring those fields keeps a
        # missing inspector or malformed child report fail-closed.
        structurally_validated = (
            isinstance(validation, Mapping)
            and validation.get("model_seen") is True
            and "errors" in validation
            and isinstance(validation.get("errors"), (list, tuple))
            and not list(validation.get("errors") or ())
            and validation.get("strict_ok") is True
        )
        pending = (
            allow_pending
            and quantized is True
            and status in {
                "qwen2_fused_fake_dequant_active",
                "qwen2_fused_fake_dequant_pending_hit",
                "pending_hit",
                "pending",
            }
            and model_counter_available
            and model_identity not in (None, "")
            and model_forward_delta == 0
            and structurally_validated
        )
        if pending:
            metadata["rollout_kv_verified"] = False
            metadata["rollout_kv_strict_pending"] = True
            metadata["rollout_kv_strict_deferred"] = True
            return
        reason = (
            f"vLLM rollout status={status!r}, quantized={quantized!r}, "
            f"model_forward_calls={hits}, model_forward_delta={model_forward_delta}; "
            "a visible validated Qwen2 model and a fresh patched forward after "
            "the rollout baseline are required"
        )
    elif normalized_name in {"hf", "huggingface", "transformers"}:
        # HF has no separate engine; sharing the actor hook is the strongest
        # proof available on this path.
        accepted = quantized is True and status in {"actor_hook_shared", "kivi_hf_shared"}
        if accepted:
            metadata["rollout_kv_verified"] = True
            metadata["rollout_kv_strict_pending"] = False
            metadata["rollout_kv_strict_deferred"] = False
            return
        reason = f"HF rollout status={status!r}, quantized={quantized!r}; actor hook is not shared"
    else:
        reason = (
            f"rollout engine={rollout_name!r} has no supported quantized-runtime integration; "
            f"status={status!r}, quantized={quantized!r}"
        )

    metadata["rollout_kv_verified"] = False
    detail = ""
    if status in _STRICT_ROLLOUT_REJECT_STATUSES or hits <= 0:
        detail = " (patch unavailable, validation/model unavailable, pending, or no-hit state)"
    raise RuntimeError(
        f"{REQUIRE_QUANTIZED_ROLLOUT_ENV}=1 requires a verified quantized rollout; "
        + reason
        + detail
    )


def _install_deferred_rollout_guard(worker: Any) -> bool:
    """Check a strict vLLM patch after the first actual generation.

    vLLM attention modules are constructed during ``_build_rollout`` but the
    patched-forward hit counter cannot become positive until
    ``generate_sequences`` is called.  Installing a small instance-level
    wrapper on the already-built rollout lets strict mode defer *only* that
    one proof obligation while still failing before the generated batch is
    returned to the trainer.  The wrapper is intentionally limited to the
    synchronous/async public generation methods exposed by the pinned rollout
  object; an object without that method is rejected rather than treated as
  verified.
    """

    if not _quantized_rollout_required():
        return False
    rollout = getattr(worker, "rollout", None)
    if rollout is None:
        raise RuntimeError(
            f"{REQUIRE_QUANTIZED_ROLLOUT_ENV}=1 cannot defer verification: "
            "rollout object is unavailable"
        )
    if bool(getattr(rollout, "_qat_kv_strict_generation_guard", False)):
        return True
    if bool(getattr(rollout, "_qat_kv_strict_generation_failed", False)):
        raise RuntimeError(
            f"{REQUIRE_QUANTIZED_ROLLOUT_ENV}=1 rollout generation guard is already failed"
        )

    method_name = "generate_sequences"
    # The pinned VeRL actor worker calls exactly this method.  Do not fall back
    # to a generic ``generate`` API: doing so could verify a helper path that
    # the trainer never uses, which would defeat the fail-closed contract.
    original = getattr(rollout, method_name, None)
    if not callable(original):
        raise RuntimeError(
            f"{REQUIRE_QUANTIZED_ROLLOUT_ENV}=1 cannot defer verification: "
            "rollout does not expose a callable generate_sequences method"
        )

    def _record_failure(exc: BaseException) -> None:
        """Persist a strict failure without hiding the original exception."""

        try:
            current = get_qat_kv_metadata(worker)
            if not current:
                current = _base_metadata(
                    resolve_settings(getattr(worker, "config", None)),
                    rollout_name=_rollout_name(getattr(worker, "config", None)),
                )
            current.update(
                {
                    "rollout_kv_verified": False,
                    "rollout_kv_strict_pending": False,
                    "rollout_kv_strict_deferred": False,
                    "rollout_kv_strict_generation_failed": True,
                    "rollout_kv_strict_generation_verified": False,
                    "rollout_kv_strict_error": f"{type(exc).__name__}: {exc}",
                    "warning": str(exc),
                }
            )
            _publish_metadata(worker, current)
            setattr(rollout, "qat_kv_metadata", dict(current))
        except Exception:
            LOGGER.debug("could not publish deferred strict rollout failure", exc_info=True)
        try:
            setattr(rollout, "_qat_kv_strict_generation_failed", True)
        except Exception:
            pass

    def _finalize() -> None:
        # A non-pending call is required here.  If the patched class was
        # imported in a child process or never executed, this raises and the
        # generated batch is not handed to the optimizer.
        _annotate_rollout(worker, allow_pending=False)
        verified = get_qat_kv_metadata(worker).get("rollout_kv_verified") is True
        if not verified:
            raise RuntimeError(
                f"{REQUIRE_QUANTIZED_ROLLOUT_ENV}=1 generation completed without "
                "verified quantized vLLM attention hits"
            )
        # Publish the post-generation state as well as the private marker.
        # Construction-time metadata is intentionally pending; only a
        # successful generation followed by a positive runtime probe may set
        # this flag.
        current = get_qat_kv_metadata(worker)
        current.update(
            {
                "rollout_kv_strict_pending": False,
                "rollout_kv_strict_deferred": False,
                "rollout_kv_strict_generation_failed": False,
                "rollout_kv_strict_generation_verified": True,
            }
        )
        _publish_metadata(worker, current)
        try:
            setattr(rollout, "qat_kv_metadata", dict(current))
        except Exception:
            LOGGER.debug("could not attach verified strict metadata to rollout", exc_info=True)

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def _guarded_async(*args: Any, **kwargs: Any) -> Any:
            if bool(getattr(rollout, "_qat_kv_strict_generation_failed", False)):
                raise RuntimeError(
                    f"{REQUIRE_QUANTIZED_ROLLOUT_ENV}=1 rollout generation guard previously failed"
                )
            if bool(getattr(rollout, "_qat_kv_strict_generation_verified", False)):
                return await original(*args, **kwargs)
            try:
                result = await original(*args, **kwargs)
                _finalize()
                setattr(rollout, "_qat_kv_strict_generation_verified", True)
                return result
            except Exception as exc:
                _record_failure(exc)
                raise

        guarded = _guarded_async
    else:

        @functools.wraps(original)
        def _guarded_sync(*args: Any, **kwargs: Any) -> Any:
            if bool(getattr(rollout, "_qat_kv_strict_generation_failed", False)):
                raise RuntimeError(
                    f"{REQUIRE_QUANTIZED_ROLLOUT_ENV}=1 rollout generation guard previously failed"
                )
            if bool(getattr(rollout, "_qat_kv_strict_generation_verified", False)):
                return original(*args, **kwargs)
            try:
                result = original(*args, **kwargs)
                if inspect.isawaitable(result):

                    async def _await_and_finalize() -> Any:
                        try:
                            awaited = await result
                            _finalize()
                            setattr(rollout, "_qat_kv_strict_generation_verified", True)
                            return awaited
                        except Exception as exc:
                            _record_failure(exc)
                            raise

                    return _await_and_finalize()
                _finalize()
                setattr(rollout, "_qat_kv_strict_generation_verified", True)
                return result
            except Exception as exc:
                _record_failure(exc)
                raise

        guarded = _guarded_sync

    try:
        setattr(rollout, method_name, guarded)
        setattr(rollout, "_qat_kv_strict_generation_guard", True)
        setattr(rollout, "_qat_kv_strict_generation_original", original)
        setattr(rollout, "_qat_kv_strict_generation_method", method_name)
    except Exception as exc:
        raise RuntimeError(
            f"{REQUIRE_QUANTIZED_ROLLOUT_ENV}=1 could not install deferred "
            "generation verification wrapper"
        ) from exc
    return True


def _annotate_rollout(worker: Any, *, allow_pending: bool = False) -> dict[str, Any]:
    """Mark rollout scope and warn when the engine bypasses HF hooks."""

    config = getattr(worker, "config", None)
    name = _rollout_name(config)
    new_probe = bool(getattr(worker, _ROLLOUT_PROBE_PENDING_ATTR, False))
    if new_probe:
        # Consume the one-shot construction marker.  Deferred generation
        # probes must retain the baseline captured below.
        try:
            setattr(worker, _ROLLOUT_PROBE_PENDING_ATTR, False)
        except Exception:
            pass
    current = get_qat_kv_metadata(worker)
    if not current:
        # This can happen for a standalone rollout worker that did not build
        # an actor.  Resolve settings only for metadata; no hook is installed.
        current = _base_metadata(resolve_settings(config), rollout_name=name, config=config)
    else:
        current["rollout_engine"] = name

    if _is_vllm_like(name):
        # The model-specific Qwen2 patch is opt-in and must have been installed
        # before the engine was built.  A status object on the worker is more
        # reliable than merely checking the environment: it lets manifests
        # distinguish an active class patch from a typo in a launcher env.
        vllm_patch = getattr(worker, "_qat_kv_vllm_patch_metadata", None)
        patch_active = isinstance(vllm_patch, Mapping) and bool(vllm_patch.get("installed"))
        if patch_active:
            # A class-level monkeypatch is only a necessary condition.  In
            # vLLM V1/async/Ray modes the actual model may be constructed in a
            # child process and therefore bypass the parent import.  Whenever
            # the rollout object exposes its model, run the structural probe
            # from ``vllm_qwen2_patch`` and publish the result.  The fallback
            # keeps compatibility with tiny downstream patch shims that only
            # return an install status (and with older rollout wrappers that
            # intentionally hide their model).
            rollout = getattr(worker, "rollout", None)
            runtime_probe: Optional[dict[str, Any]] = None
            if rollout is not None:
                try:
                    patch_module = importlib.import_module("qat_kv.vllm_qwen2_patch")
                    inspector = getattr(patch_module, "rollout_metadata", None)
                    if callable(inspector):
                        runtime_probe = dict(inspector(rollout))
                except Exception as exc:
                    # Diagnostics must not mask a successful rollout build;
                    # retain the install metadata and make the uncertainty
                    # explicit below.
                    runtime_probe = {
                        "probe_error": f"{type(exc).__name__}: {exc}",
                    }

            patch_report = dict(vllm_patch)
            if runtime_probe is not None:
                patch_report["runtime_probe"] = runtime_probe
            _record_vllm_model_probe(
                current,
                runtime_probe,
                worker,
                new_probe=new_probe,
            )
            validation = (
                runtime_probe.get("model_validation")
                if isinstance(runtime_probe, Mapping)
                else None
            )
            if isinstance(validation, Mapping) and bool(validation.get("model_seen")):
                validation_errors = list(validation.get("errors") or ())
                if validation_errors:
                    current.update(
                        {
                            "rollout_kv_quantized": False,
                            "rollout_kv_status": "qwen2_fused_fake_dequant_validation_failed",
                            "rollout_policy_mismatch": True,
                            "vllm_kv_patch": patch_report,
                            "warning": (
                                "Qwen2 fused-QKV patch was requested, but the constructed "
                                "vLLM model failed patch validation: "
                                + ", ".join(str(item) for item in validation_errors)
                            ),
                        }
                    )
                    LOGGER.error("%s: vLLM KV patch validation failed: %s", PLUGIN_NAME, validation_errors)
                else:
                    # Treat malformed/missing counters as zero so the
                    # resulting state is the explicit pending/no-hit status;
                    # strict mode will then reject it with a useful reason.
                    hits = _vllm_rollout_hits({"vllm_kv_patch": patch_report})
                    current.update(
                        {
                            "rollout_kv_quantized": True,
                            "rollout_kv_status": (
                                "qwen2_fused_fake_dequant_active"
                                if hits > 0
                                else "qwen2_fused_fake_dequant_pending_hit"
                            ),
                            "rollout_policy_mismatch": False,
                            "vllm_kv_patch": patch_report,
                            "warning": (
                                "Qwen2 fused-QKV KV fake/dequant patch is present for TP=1 "
                                "external_launcher; cache remains floating point and exact OPD "
                                "parity still requires a same-sequence check."
                            ),
                        }
                    )
            elif isinstance(validation, Mapping) and not bool(validation.get("model_seen")):
                # Do not claim exact policy parity when the engine hides its
                # model (or moved it into a child process).  The class patch
                # may still be active, so retain a distinct pending state.
                current.update(
                    {
                        "rollout_kv_quantized": True,
                        "rollout_kv_status": "qwen2_fused_fake_dequant_model_unavailable",
                        "rollout_policy_mismatch": None,
                        "vllm_kv_patch": patch_report,
                        "warning": (
                            "Qwen2 fused-QKV patch was installed, but the vLLM model could not "
                            "be inspected; rollout quantization remains unverified."
                        ),
                    }
                )
            else:
                current.update(
                    {
                        "rollout_kv_quantized": True,
                        "rollout_kv_status": "qwen2_fused_fake_dequant_pending_hit",
                        "rollout_policy_mismatch": False,
                        "vllm_kv_patch": patch_report,
                        "warning": (
                            "Qwen2 fused-QKV KV fake/dequant patch installed for TP=1 "
                            "external_launcher; cache remains floating point and exact OPD "
                            "parity still requires a same-sequence check."
                        ),
                    }
                )
        else:
            # A failed/unrequested patch has no model-scoped evidence.  Clear
            # any prior rollout snapshot so a subsequent strict check cannot
            # inherit it.
            if new_probe:
                _record_vllm_model_probe(current, None, worker, new_probe=True)
            failed_patch = getattr(worker, "_qat_kv_vllm_patch_metadata", None)
            failed_patch_report = (
                dict(failed_patch) if isinstance(failed_patch, Mapping) else None
            )
            failure_reason = (
                str(failed_patch_report.get("reason"))
                if failed_patch_report and failed_patch_report.get("reason")
                else None
            )
            current.update(
                {
                    "rollout_kv_quantized": False,
                    "rollout_kv_status": "unquantized_external_engine",
                    "rollout_policy_mismatch": True,
                    "vllm_kv_patch": failed_patch_report,
                    "warning": (
                        (
                            "vLLM KV patch was requested but failed ("
                            + failure_reason
                            + "); rollout KV is unquantized."
                        )
                        if failure_reason
                        else (
                            "vLLM-like rollout does not execute Hugging Face k_proj/v_proj hooks; "
                            "rollout KV is unquantized. Actor fake-dequant scope is not exact OPD."
                        )
                    ),
                }
            )
            warned = bool(getattr(worker, "_qat_kv_rollout_warned", False))
            if not warned:
                message = (
                    f"{PLUGIN_NAME}: {name} rollout bypasses HF KV fake-quant hooks; "
                    "rollout_kv_quantized=False (actor fake-dequant scope is not exact OPD)"
                )
                LOGGER.warning(message)
                warnings.warn(message, RuntimeWarning, stacklevel=3)
                setattr(worker, "_qat_kv_rollout_warned", True)
    elif _is_external_rollout(name):
        current.update(
            {
                "rollout_kv_quantized": False,
                "rollout_kv_status": "unquantized_external_engine",
                "rollout_policy_mismatch": True,
            }
        )
    elif name is not None:
        # An HF rollout may share the actor module, but we only claim it is
        # quantized if the actor hook was actually installed.
        actor_quantized = bool(current.get("actor_kv_fake_quantized", False))
        current.update(
            {
                "rollout_kv_quantized": actor_quantized,
                "rollout_kv_status": (
                    "kivi_hf_shared"
                    if actor_quantized and bool(current.get("kivi_hf_rollout"))
                    else "actor_hook_shared"
                    if actor_quantized
                    else "unquantized"
                ),
                "rollout_policy_mismatch": not actor_quantized,
            }
        )

    # A successful class install is not itself runtime evidence.  Record the
    # verified bit only for an active/hit vLLM probe (or a shared HF actor
    # hook), then apply the optional strict gate before publishing the final
    # worker metadata.
    current["rollout_kv_verified"] = bool(
        (
            current.get("rollout_kv_quantized") is True
            and current.get("rollout_kv_status") == "qwen2_fused_fake_dequant_active"
            and current.get("rollout_kv_model_counter_available") is True
            and current.get("rollout_kv_model_identity") not in (None, "")
            and _vllm_rollout_forward_delta(current) is not None
            and _vllm_rollout_forward_delta(current) > 0
        )
        or (
            current.get("rollout_kv_quantized") is True
            and current.get("rollout_kv_status") in {"actor_hook_shared", "kivi_hf_shared"}
        )
    )
    rollout_model = getattr(worker, "actor_module_fsdp", None)
    if rollout_model is None:
        rollout_model = getattr(worker, "actor_module", None)
    rollout = getattr(worker, "rollout", None)
    try:
        _enforce_quantized_rollout(
            current,
            rollout_name=name,
            allow_pending=allow_pending,
        )
        if current.get("rollout_kv_strict_pending"):
            _install_deferred_rollout_guard(worker)
    except Exception as exc:
        # Keep the strict failure visible on the Ray worker before propagating
        # it.  This is especially useful when the original rollout object is
        # subsequently cleaned up by VeRL.
        current["rollout_kv_strict_error"] = f"{type(exc).__name__}: {exc}"
        current["warning"] = str(exc)
        _publish_metadata(worker, current, model=rollout_model)
        if rollout is not None:
            try:
                setattr(rollout, "qat_kv_metadata", dict(current))
            except Exception:
                LOGGER.debug("could not attach strict KV failure metadata to rollout", exc_info=True)
        raise

    result = _publish_metadata(worker, current, model=rollout_model)
    if rollout is not None:
        try:
            setattr(rollout, "qat_kv_metadata", dict(result))
        except Exception:
            LOGGER.debug("could not attach KV metadata to rollout object", exc_info=True)
    return result


def _flash_attn_available() -> bool:
    """Return whether Transformers can actually use optional flash-attn."""

    try:
        from transformers.utils import is_flash_attn_2_available

        return bool(is_flash_attn_2_available())
    except (ImportError, ModuleNotFoundError, AttributeError):
        # Older Transformers releases do not expose the helper; fall back to a
        # package-spec check there.
        try:
            return importlib.util.find_spec("flash_attn") is not None
        except (ImportError, ModuleNotFoundError, ValueError, OSError, RuntimeError):
            return False
    except ValueError:
        # ``find_spec`` raises ValueError for a broken module already present in
        # sys.modules; treating that as unavailable is the safe choice.
        return False
    except Exception:
        # A binary/ABI failure while probing flash-attn must also select eager;
        # allowing it to propagate would recreate the startup failure we are
        # trying to avoid.
        return False


def _reward_attention_settings(worker: Any) -> dict[str, Any]:
    """Resolve the reward-worker attention backend and fallback policy.

    The Open-MOPD FSDP reward worker currently passes
    ``attn_implementation='flash_attention_2'`` unconditionally.  In the
    pinned environment flash-attn is not installed, so ``auto`` (the default)
    selects eager attention.  An explicit ``eager``/``sdpa`` request is also
    honoured.  A request for flash attention falls back to eager when the
    optional package is unavailable, preventing a startup crash.
    """

    config = getattr(worker, "config", None)
    requested, _ = _first_config_value(
        config,
        (
            ("reward_attention",),
            ("model", "reward_attention"),
            ("model", "override_config", "attn_implementation"),
            ("model", "attn_implementation"),
            ("attn_implementation",),
        ),
    )
    env_requested = os.environ.get(REWARD_ATTN_ENV, _MISSING)
    if env_requested is not _MISSING and env_requested is not None:
        requested = env_requested
    if requested is _MISSING or requested is None:
        requested = "auto"
    requested = str(requested).strip().lower()
    aliases = {"flash": "flash_attention_2", "flash_attn": "flash_attention_2"}
    requested = aliases.get(requested, requested)
    if requested not in {"auto", "eager", "sdpa", "flash_attention_2"}:
        raise ValueError(
            f"{REWARD_ATTN_ENV} must be one of auto,eager,sdpa,flash_attention_2; got {requested!r}"
        )

    available = _flash_attn_available()
    if requested == "auto":
        selected = "flash_attention_2" if available else "eager"
        reason = "auto_detected" if available else "flash_attn_unavailable"
    elif requested == "flash_attention_2" and not available:
        selected = "eager"
        reason = "requested_flash_attn_unavailable"
    else:
        selected = requested
        reason = "explicit"
    return {
        "requested": requested,
        "selected": selected,
        "flash_attn_available": available,
        "force_eager": selected == "eager",
        "reason": reason,
    }


@contextmanager
def _temporary_reward_attention_backend(backend: str):
    """Temporarily rewrite AutoModelForCausalLM's backend kwarg.

    The reference reward worker imports ``AutoModelForCausalLM`` inside
    ``_build_model`` and hardcodes the kwarg.  Patching the class method for the
    duration of that one call is considerably smaller and safer than copying
    the whole worker implementation.  A process-local lock prevents concurrent
    reward builds from observing the temporary method.
    """

    if backend == "flash_attention_2":
        yield
        return

    from transformers import AutoModelForCausalLM

    owner = None
    descriptor = None
    for candidate in AutoModelForCausalLM.__mro__:
        if "from_pretrained" in candidate.__dict__:
            owner = candidate
            descriptor = candidate.__dict__["from_pretrained"]
            break
    if owner is None or descriptor is None:
        raise RuntimeError("could not locate AutoModelForCausalLM.from_pretrained for eager fallback")

    original_bound = AutoModelForCausalLM.from_pretrained

    if isinstance(descriptor, classmethod):

        def _patched(_cls, *args, **kwargs):
            call_kwargs = dict(kwargs)
            call_kwargs["attn_implementation"] = backend
            return original_bound(*args, **call_kwargs)

        replacement = classmethod(_patched)
    else:

        def _patched(*args, **kwargs):
            call_kwargs = dict(kwargs)
            call_kwargs["attn_implementation"] = backend
            return original_bound(*args, **call_kwargs)

        replacement = _patched

    with _REWARD_PATCH_LOCK:
        setattr(owner, "from_pretrained", replacement)
        try:
            yield
        finally:
            setattr(owner, "from_pretrained", descriptor)


def _publish_reward_metadata(worker: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    try:
        setattr(worker, "qat_kv_reward_metadata", result)
        setattr(worker, "_qat_kv_reward_metadata", result)
    except Exception:
        LOGGER.debug("could not attach reward attention metadata", exc_info=True)
    return result


def _publish_reward_model_metadata(model: Any, metadata: Mapping[str, Any]) -> None:
    if model is None:
        return
    try:
        setattr(model, "qat_kv_reward_metadata", dict(metadata))
    except Exception:
        LOGGER.debug("could not attach reward metadata to model", exc_info=True)


def _reward_build_hardcodes_flash(build_method: Any) -> bool:
    """Detect the legacy literal flash-attention call without importing source.

    Open-MOPD's historical worker contains
    ``attn_implementation="flash_attention_2"`` inside ``_build_model``;
    patched forks read the value from config instead.  If source inspection is
    unavailable (for example after cloudpickle), assume the legacy form so a
    missing flash-attn package still fails closed to eager.
    """

    try:
        source = inspect.getsource(inspect.unwrap(build_method))
    except (OSError, TypeError):
        return True
    import re

    return re.search(r"attn_implementation\s*=\s*['\"]flash_attention_2['\"]", source) is not None


def patch_reward_worker_class(worker_class: type) -> bool:
    """Patch one FSDP ``RewardModelWorker`` for flash-attn-free startup."""

    if not isinstance(worker_class, type) or worker_class.__dict__.get(_REWARD_PATCH_MARKER, False):
        return False
    original_build = getattr(worker_class, "_build_model", None)
    if not callable(original_build):
        return False

    @functools.wraps(original_build)
    def _build_wrapper(self, *args, **kwargs):
        settings = _reward_attention_settings(self)
        legacy_hardcoded = _reward_build_hardcodes_flash(original_build)
        explicit_override = os.environ.get(REWARD_ATTN_ENV, _MISSING) is not _MISSING
        needs_backend_override = settings["selected"] != "flash_attention_2" and (
            legacy_hardcoded or explicit_override or settings["requested"] == "flash_attention_2"
        )
        metadata = {
            "plugin": PLUGIN_NAME,
            "plugin_version": PLUGIN_VERSION,
            "scope": "reward_model",
            "exact_opd": False,
            "packed_kv_cache": False,
            "kv_hook_applied": False,
            "reward_model_kv_bits": 32,
            "reward_model_kv_fake_quantized": False,
            "teacher_untouched": True,
            "reward_model_untouched": True,
            "requested_attention_implementation": settings["requested"],
            "selected_attention_implementation": settings["selected"],
            "flash_attn_available": settings["flash_attn_available"],
            "fallback_reason": settings["reason"],
            "legacy_hardcoded_flash": legacy_hardcoded,
            "temporary_override_applied": needs_backend_override,
            "status": "building",
        }
        _publish_reward_metadata(self, metadata)
        try:
            if not needs_backend_override:
                result = original_build(self, *args, **kwargs)
            else:
                with _temporary_reward_attention_backend(settings["selected"]):
                    result = original_build(self, *args, **kwargs)
        except Exception:
            metadata["status"] = "build_failed"
            _publish_reward_metadata(self, metadata)
            raise
        metadata["status"] = "built"
        _publish_reward_metadata(self, metadata)
        _publish_reward_model_metadata(result, metadata)
        if settings["force_eager"]:
            LOGGER.warning(
                "%s: reward worker requested %s; using eager attention (%s)",
                PLUGIN_NAME,
                settings["requested"],
                settings["reason"],
            )
        return result

    worker_class._build_model = _build_wrapper
    setattr(worker_class, "__qat_kv_original_build_model__", original_build)
    setattr(worker_class, _REWARD_PATCH_MARKER, True)
    return True


def _patch_kivi_apply_monkey_patch(worker_module: Any) -> bool:
    """Install KIVI immediately after VeRL's model monkey patches.

    The pinned FSDP worker calls ``apply_monkey_patch`` after constructing the
    actor and before wrapping it with FSDP.  Wrapping that imported symbol is
    the narrowest way to place KIVI at the correct lifecycle point without
    copying VeRL's model builder or changing its optimizer/checkpoint flow.
    """

    if not _kivi_hf_enabled():
        return False
    original = getattr(worker_module, "apply_monkey_patch", None)
    if not callable(original) or getattr(original, _KIVI_APPLY_PATCH_MARKER, False):
        return False

    @functools.wraps(original)
    def _wrapped(model: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(model, *args, **kwargs)
        if _KIVI_ACTOR_BUILD_CONTEXT.get():
            config = _KIVI_CONFIG_CONTEXT.get()
            rollout_name = _rollout_name(config)
            if rollout_name not in {"hf", "huggingface", "transformers", None}:
                raise RuntimeError(
                    f"{KIVI_HF_ENABLE_ENV}=1 requires rollout.name=hf; got {rollout_name!r}"
                )
            bits, group_size, residual_length = _kivi_settings(config)
            from .kivi_qwen2 import install_kivi_hf_rollout

            install_kivi_hf_rollout(
                model,
                bits=bits,
                group_size=group_size,
                residual_length=residual_length,
                training_fake_quant=True,
            )
        return result

    setattr(_wrapped, _KIVI_APPLY_PATCH_MARKER, True)
    worker_module.apply_monkey_patch = _wrapped
    return True


def _extract_role(original: Any, self: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
    role = kwargs.get("role", _MISSING)
    if role is _MISSING:
        try:
            signature = inspect.signature(original)
            bound = signature.bind_partial(self, *args, **kwargs)
            role = bound.arguments.get("role", _MISSING)
            # ``Signature.bind_partial`` omits parameters that were not
            # supplied, even when they have a meaningful default.  The
            # reference FSDP builder defaults to ``role="actor"``; honour
            # that default instead of falling back to the worker's composite
            # role name (for example ``actor_rollout_ref``).
            if role is _MISSING:
                role_parameter = signature.parameters.get("role")
                if role_parameter is not None and role_parameter.default is not inspect.Parameter.empty:
                    role = role_parameter.default
        except (TypeError, ValueError):
            role = _MISSING
    if role is _MISSING or role is None:
        role = getattr(self, "role", "actor")
    normalized = str(role).lower()
    # A few forks use the worker's composite role as the builder argument (or
    # omit the argument entirely).  Only the actor-bearing composite roles are
    # normalized here; a standalone rollout/ref worker remains untouched via
    # its ``_is_actor`` guard in ``patch_worker_class``.
    if normalized in {"actor_rollout", "actor_rollout_ref"}:
        return "actor"
    return normalized


def patch_worker_class(worker_class: type) -> bool:
    """Patch one VeRL ``ActorRolloutRefWorker`` class.

    Returns ``True`` when at least one method was newly wrapped.  This public
    function is also useful for tests and for VeRL forks whose worker module is
    imported after this external module.
    """

    if not isinstance(worker_class, type):
        return False
    # Inspect ``__dict__`` rather than ``getattr`` so a subclass of a patched
    # base worker can still be wrapped when it overrides model construction.
    if worker_class.__dict__.get(_PATCH_MARKER, False):
        return False

    patched = False
    original_build = getattr(worker_class, "_build_model_optimizer", None)
    if callable(original_build):

        @functools.wraps(original_build)
        def _build_wrapper(self, *args, **kwargs):
            role = _extract_role(original_build, self, args, kwargs)
            actor_token = _KIVI_ACTOR_BUILD_CONTEXT.set(
                role == "actor" and bool(getattr(self, "_is_actor", True))
            )
            config_token = _KIVI_CONFIG_CONTEXT.set(getattr(self, "config", None))
            try:
                result = original_build(self, *args, **kwargs)
            finally:
                _KIVI_CONFIG_CONTEXT.reset(config_token)
                _KIVI_ACTOR_BUILD_CONTEXT.reset(actor_token)
            # ``role=actor`` is the only path that creates trainable actor
            # weights.  In particular, role=ref is the frozen teacher path.
            if role == "actor" and bool(getattr(self, "_is_actor", True)):
                model = _model_from_build_result(result)
                if model is None:
                    model = getattr(self, "actor_module_fsdp", None)
                    if model is None:
                        model = getattr(self, "actor_module", None)
                _attach_actor_model(self, model)
            elif role == "ref":
                # A ref/teacher object must be distinct from the hooked actor;
                # aliasing would otherwise make the teacher quantized by
                # accident.  Check common FSDP/PEFT wrapper links as well as
                # the outer object, then refuse the unsafe construction.
                ref_model = _model_from_build_result(result)
                actor_ids = {
                    value
                    for value in (
                        getattr(self, "_qat_kv_actor_model_id", None),
                        getattr(self, "_qat_kv_hook_target_id", None),
                    )
                    if value is not None
                }
                ref_ids = {id(candidate) for _path, candidate in _iter_model_candidates(ref_model)}
                if actor_ids.intersection(ref_ids):
                    raise RuntimeError("reference/teacher model aliases the hooked actor; refusing to quantize teacher")
            return result

        worker_class._build_model_optimizer = _build_wrapper
        setattr(worker_class, "__qat_kv_original_build_model_optimizer__", original_build)
        patched = True

    # A few VeRL forks hide model construction behind init_model.  Keep a
    # narrow fallback; it only runs when no actor hook was installed by the
    # dedicated builder wrapper.
    original_init = getattr(worker_class, "init_model", None)
    if callable(original_init):

        @functools.wraps(original_init)
        def _init_wrapper(self, *args, **kwargs):
            actor_token = _KIVI_ACTOR_BUILD_CONTEXT.set(bool(getattr(self, "_is_actor", False)))
            config_token = _KIVI_CONFIG_CONTEXT.set(getattr(self, "config", None))
            try:
                result = original_init(self, *args, **kwargs)
            finally:
                _KIVI_CONFIG_CONTEXT.reset(config_token)
                _KIVI_ACTOR_BUILD_CONTEXT.reset(actor_token)
            if bool(getattr(self, "_is_actor", False)) and not getattr(self, "_qat_kv_hook", None):
                model = getattr(self, "actor_module_fsdp", None)
                if model is None:
                    model = getattr(self, "actor_module", None)
                if model is not None:
                    _attach_actor_model(self, model)
            return result

        worker_class.init_model = _init_wrapper
        setattr(worker_class, "__qat_kv_original_init_model__", original_init)
        patched = True

    original_rollout = getattr(worker_class, "_build_rollout", None)
    if callable(original_rollout):

        @functools.wraps(original_rollout)
        def _rollout_wrapper(self, *args, **kwargs):
            # Install the optional fused-QKV patch *before* the original
            # builder imports/constructs vLLM.  If the opt-in is malformed or
            # unsupported, fail closed rather than silently falling back to an
            # unquantized rollout.  With the default environment this is a
            # no-op and the existing explicit mismatch warning is retained.
            # Mark a fresh probe before construction so a reused vLLM model
            # gets a new model-counter baseline in the ``finally`` annotation.
            probe_counter = _safe_nonnegative_int(
                getattr(self, _ROLLOUT_PROBE_COUNTER_ATTR, 0)
            ) or 0
            try:
                setattr(self, _ROLLOUT_PROBE_COUNTER_ATTR, probe_counter + 1)
                setattr(self, _ROLLOUT_PROBE_ID_ATTR, probe_counter + 1)
                setattr(self, _ROLLOUT_PROBE_PENDING_ATTR, True)
            except Exception:
                # Metadata is best effort; strict mode will still reject if
                # the model-scoped baseline cannot be recorded.
                LOGGER.debug("could not mark vLLM rollout probe", exc_info=True)
            install_completed = False
            try:
                rollout_name = _rollout_name(getattr(self, "config", None))
                _install_vllm_rollout_patch(self, rollout_name)
                install_completed = True
                # The pinned worker constructs rollouts without passing the
                # actor module.  Provide it only during construction so the
                # KIVI HF adapter can reuse VeRL's actor instance.
                kivi_module_token = None
                if rollout_name in {"hf", "huggingface", "transformers", None} and _kivi_hf_enabled():
                    from .kivi_qwen2 import set_kivi_rollout_module

                    actor_module = getattr(self, "actor_module_fsdp", None)
                    if actor_module is None:
                        actor_module = getattr(self, "actor_module", None)
                    kivi_module_token = set_kivi_rollout_module(actor_module)
                try:
                    return original_rollout(self, *args, **kwargs)
                finally:
                    if kivi_module_token is not None:
                        from .kivi_qwen2 import reset_kivi_rollout_module

                        reset_kivi_rollout_module(kivi_module_token)
            except Exception as exc:
                # This includes malformed settings, an unsupported backend,
                # and actor/rollout policy mismatches.  Record it before the
                # exception crosses the Ray boundary so manifests retain the
                # fail-closed reason.
                try:
                    patch_requested = _vllm_enable_requested()
                except Exception:
                    # A malformed opt-in is itself a requested failure; do
                    # not let the diagnostic probe replace the root error.
                    patch_requested = True
                # Once the installer returned successfully, an exception is
                # a rollout-construction/runtime failure rather than an
                # installer failure; retain the active patch metadata so the
                # diagnosis does not mislabel the root cause.
                if patch_requested and not install_completed:
                    _record_vllm_patch_failure(self, exc)
                raise
            finally:
                # ``finally`` ensures a failed rollout build still leaves an
                # honest diagnostic in the worker object/log.
                active_exception = sys.exc_info()[1]
                try:
                    # A successful build may legitimately have zero patched
                    # forwards so far.  Defer only that pending-hit proof to
                    # the first generation call; a failed builder is never
                    # allowed to turn into a deferred/ambiguous state.
                    _annotate_rollout(self, allow_pending=active_exception is None)
                except Exception:
                    # A strict runtime policy is itself part of the rollout
                    # contract and must cross the Ray boundary.  If the
                    # original builder already failed, preserve that primary
                    # exception and only log the secondary annotation issue.
                    try:
                        strict_requested = _quantized_rollout_required()
                    except Exception:
                        # A malformed strict flag is a fail-closed request.
                        strict_requested = True
                    if active_exception is None and strict_requested:
                        raise
                    LOGGER.exception("%s: could not annotate rollout metadata", PLUGIN_NAME)

        worker_class._build_rollout = _rollout_wrapper
        setattr(worker_class, "__qat_kv_original_build_rollout__", original_rollout)
        patched = True

    if patched:
        setattr(worker_class, _PATCH_MARKER, True)
    return patched


def install_worker_patches() -> tuple[str, ...]:
    """Best-effort patch of VeRL worker classes available in this process."""

    global PATCHED_WORKERS
    patched: list[str] = []
    # The pinned Open-MOPD checkout ships HFRollout but does not register it.
    # Register that existing VeRL rollout class here; no generation or RL logic
    # is implemented by this plugin.
    if _kivi_hf_enabled():
        try:
            base = importlib.import_module("verl.workers.rollout.base")
            registry = getattr(base, "_ROLLOUT_REGISTRY", None)
            if isinstance(registry, dict):
                registry[("hf", "sync")] = "qat_kv.kivi_qwen2.KIVIHFRollout"
        except (ImportError, ModuleNotFoundError):
            pass
    # FSDP is the supported Open-MOPD path.  Do not import optional Megatron or
    # role modules just for discovery (they can have heavyweight/side-effectful
    # imports); patch them only when a caller has already loaded one explicitly.
    module_names = ["verl.workers.fsdp_workers"]
    module_names.extend(
        name
        for name in ("verl.workers.megatron_workers", "verl.workers.roles")
        if name in sys.modules
    )
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            continue
        except Exception:
            # External-module import should not obscure VeRL's own import
            # diagnostics.  The worker calls this function again in its
            # process, and a direct ``patch_worker_class`` remains available.
            LOGGER.debug("could not inspect %s for KV plugin patch", module_name, exc_info=True)
            continue
        if _patch_kivi_apply_monkey_patch(module):
            patched.append(f"{module_name}.apply_monkey_patch(kivi-hf)")
        worker_class = getattr(module, "ActorRolloutRefWorker", None)
        if worker_class is not None and patch_worker_class(worker_class):
            patched.append(f"{module_name}.ActorRolloutRefWorker")
        reward_class = getattr(module, "RewardModelWorker", None)
        if reward_class is not None and patch_reward_worker_class(reward_class):
            patched.append(f"{module_name}.RewardModelWorker(attn=eager-fallback)")
    newly_patched = tuple(patched)
    # Keep the module-level diagnostic cumulative across an explicit
    # post-import install (the second call itself still reports only newly
    # wrapped classes, preserving the idempotent function contract).
    previous = globals().get("PATCHED_WORKERS", ())
    PATCHED_WORKERS = tuple(dict.fromkeys((*previous, *newly_patched)))
    return newly_patched


def install() -> tuple[str, ...]:
    """Explicit alias for custom launchers that import external modules early."""

    return install_worker_patches()


def _in_worker_import_context() -> bool:
    """Whether this import is happening in a VeRL worker, not the driver.

    ``model.external_lib`` is also imported while the Hydra configuration is
    built in the driver.  Importing/patching VeRL worker classes there is an
    unnecessary global side effect.  VeRL's normal worker path imports the
    external module from ``init_model``; checking the short Python stack catches
    that path without importing Ray (``ray.get_runtime_context`` can start a
    local Ray instance as a side effect).  The Ray worker mode and torchrun
    environment checks cover versions that import external modules at worker
    startup instead.
    """

    # Explicit opt-in is useful for a custom non-Ray worker launcher and keeps
    # the default driver path safe.
    forced = os.environ.get("QAT_KV_FORCE_WORKER_PATCH", "")
    if forced:
        try:
            return _parse_bool(forced, label="QAT_KV_FORCE_WORKER_PATCH", default=False)
        except ValueError:
            # Invalid opt-in must not accidentally enable a patch.
            LOGGER.warning("ignoring invalid QAT_KV_FORCE_WORKER_PATCH=%r", forced)
            return False

    # Avoid importing ray itself here.  Its private global worker object is
    # already present when running inside a Ray actor; consulting sys.modules
    # avoids starting/importing a local Ray cluster in the driver.
    try:
        ray_worker = sys.modules.get("ray._private.worker")
        if ray_worker is None:
            raise LookupError
        mode = getattr(getattr(ray_worker, "global_worker", None), "mode", None)
        worker_mode = getattr(ray_worker, "WORKER_MODE", object())
        if mode == worker_mode:
            return True
    except Exception:
        pass

    # Standard VeRL imports happen from an init_model method.  Limit the check
    # to frames whose file path belongs to a VeRL package to avoid matching an
    # unrelated user function called ``init_model``.
    try:
        for frame in inspect.stack(context=0):
            filename = frame.filename.replace("\\", "/")
            if frame.function == "init_model" and "/verl/" in filename:
                return True
    except Exception:
        pass

    # Non-Ray torchrun workers normally have LOCAL_RANK and WORLD_SIZE set;
    # requiring a multi-process world avoids treating a plain driver with
    # RANK=0 as a worker by accident.
    if os.environ.get("LOCAL_RANK") is not None:
        try:
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                return True
        except ValueError:
            return False
    return False


# VeRL imports external_lib in both its driver/config process and worker
# processes.  Only auto-patch in the latter; callers with a custom launcher can
# invoke ``install_worker_patches()`` explicitly.
PATCHED_WORKERS = install_worker_patches() if _in_worker_import_context() else ()


__all__ = [
    "BITS_ENV",
    "EXACT_BIAS_FREE_ENV",
    "KVPluginSettings",
    "PATCHED_WORKERS",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "REQUIRE_QUANTIZED_ROLLOUT_ENV",
    "get_qat_kv_metadata",
    "install",
    "install_worker_patches",
    "patch_reward_worker_class",
    "patch_worker_class",
    "resolve_kv_bits",
    "resolve_settings",
]
