"""Resolve and attach the TurboQuant QAT scheme for a VeRL actor worker.

Kept out of ``verl_plugin`` deliberately: that module is large and its
projection-hook path is the *legacy* symmetric/KIVI quantizer.  TurboQuant is a
different operator at a different point in the network (post-RoPE, inside
attention -- see ``turboquant_attn_hook``), so it gets its own resolution and
attach path and touches ``verl_plugin`` in exactly one place.

Selection
---------
``QAT_KV_SCHEME`` is authoritative when present; otherwise the config keys
``qat_kv.scheme`` / ``quantization.kv_scheme`` / ``mopd.kv_scheme`` are used.
Accepted values are ``symmetric`` (the legacy projection hook, i.e. "not
TurboQuant") and any TurboQuant preset name.

The scheme name must match the rollout's ``kv_cache_dtype``.  Training against
``turboquant_k3v4_nc`` while generating with ``turboquant_3bit_nc`` is a silent
mismatch of exactly the kind this whole effort exists to prevent, so
``assert_matches_rollout_dtype`` makes it fatal.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

LOGGER = logging.getLogger(__name__)

SCHEME_ENV = "QAT_KV_SCHEME"
SYMMETRIC = "symmetric"

_CONFIG_PATHS = (
    ("qat_kv", "scheme"),
    ("quantization", "kv_scheme"),
    ("mopd", "kv_scheme"),
    ("qat_kv_scheme",),
    ("kv_scheme",),
)

__all__ = [
    "SCHEME_ENV",
    "SYMMETRIC",
    "assert_matches_rollout_dtype",
    "attach_turboquant_actor",
    "ensure_actor_quantizer",
    "find_hf_model",
    "is_quantized_scheme",
    "is_turboquant_scheme",
    "resolve_scheme",
]


def is_turboquant_scheme(scheme: Optional[str]) -> bool:
    return bool(scheme) and str(scheme).startswith("turboquant_")


def is_quantized_scheme(scheme: Optional[str]) -> bool:
    """True for any scheme this module installs a post-RoPE fake quantizer for.

    ``is_turboquant_scheme`` alone is not that test any more: ``int4_per_token_head``
    uses the same interception point and the same hook, but its name is not a
    TurboQuant preset.  Every gate that decides "do we quantize" must use this
    predicate -- in particular ``assert_matches_rollout_dtype``, where an
    early return for int4 would reintroduce exactly the silent train/rollout
    mismatch that check exists to catch.
    """
    from opqad.quantization.int4_ptkh_fake_quant import is_int4_ptkh_scheme
    from opqad.quantization.oscar_fake_quant import is_oscar_scheme

    return (
        is_turboquant_scheme(scheme)
        or is_int4_ptkh_scheme(scheme)
        or is_oscar_scheme(scheme)
    )


def resolve_scheme(
    config: Any = None, *, environ: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Return the requested scheme name, or ``None`` when unset.

    Raises on an unknown name rather than falling back: a typo'd preset must
    not silently degrade to the legacy quantizer.
    """
    from opqad.quantization.int4_ptkh_fake_quant import INT4_PTKH_PRESETS
    from opqad.quantization.oscar_fake_quant import OSCAR_PRESETS
    from opqad.quantization.turboquant_fake_quant import TQ_FAKE_QUANT_PRESETS

    known = set(TQ_FAKE_QUANT_PRESETS) | set(INT4_PTKH_PRESETS) | set(OSCAR_PRESETS)

    env = os.environ if environ is None else environ
    raw = env.get(SCHEME_ENV)
    source = SCHEME_ENV
    if raw is None:
        try:
            from opqad.integrations.verl_plugin import _first_config_value, _MISSING

            value, where = _first_config_value(config, _CONFIG_PATHS)
            if value is not _MISSING and value is not None:
                raw, source = str(value), where
        except Exception:  # pragma: no cover - config shapes vary
            LOGGER.debug("could not read kv scheme from config", exc_info=True)

    if raw is None:
        return None
    name = str(raw).strip()
    if not name:
        return None
    if name == SYMMETRIC:
        return SYMMETRIC
    if name not in known:
        valid = ", ".join([SYMMETRIC] + sorted(known))
        raise ValueError(f"unknown KV scheme {name!r} from {source}; valid: {valid}")
    return name


def assert_matches_rollout_dtype(scheme: str, rollout_kv_cache_dtype: Optional[str]) -> None:
    """Fail loudly when the training quantizer differs from the rollout one."""
    if not is_quantized_scheme(scheme):
        return
    if not rollout_kv_cache_dtype:
        raise ValueError(
            f"QAT scheme {scheme!r} was requested but the rollout reports no "
            "kv_cache_dtype; training would fake-quantize against an unknown "
            "inference quantizer"
        )
    if str(rollout_kv_cache_dtype) != scheme:
        raise ValueError(
            f"QAT scheme {scheme!r} does not match rollout kv_cache_dtype "
            f"{rollout_kv_cache_dtype!r}; the student would be trained for a "
            "different quantizer than it is deployed with"
        )


def attach_turboquant_actor(worker: Any, model: Any, scheme: str) -> dict[str, Any]:
    """Install the post-RoPE TurboQuant fake quantizer on one actor model."""
    from opqad.integrations.turboquant_attn_hook import install_turboquant_fake_quant
    from opqad.integrations.verl_plugin import PLUGIN_NAME, PLUGIN_VERSION

    state = install_turboquant_fake_quant(model, scheme)
    md = state.metadata()
    metadata: dict[str, Any] = {
        "plugin": PLUGIN_NAME,
        "plugin_version": PLUGIN_VERSION,
        "worker_role": str(getattr(worker, "role", "actor")),
        "scheme": scheme,
        "status": f"enabled_{scheme}",
        "actor_kv_status": f"enabled_{scheme}",
        "actor_kv_fake_quantized": True,
        "actor_kv_quantized": True,
        "actor_kv_hook_applied": True,
        "kv_hook_applied": True,
        # The legacy projection hook is explicitly NOT installed: TurboQuant
        # must quantize post-RoPE, and k_proj/v_proj outputs are pre-RoPE.
        "projection_hook_applied": False,
        "turboquant": md,
    }
    metadata.update(
        {
            "key_quant_bits": md["key_quant_bits"],
            "value_quant_bits": md["value_quant_bits"],
            "skip_layers": md["skip_layers"],
            "hook_point": md["hook_point"],
        }
    )
    setattr(worker, "_qat_turboquant_state", state)
    if int(getattr(worker, "rank", 0)) == 0:
        LOGGER.info(
            "%s: actor TurboQuant QAT installed: scheme=%s k=%d v=%d skip=%s point=%s",
            PLUGIN_NAME, scheme, md["key_quant_bits"], md["value_quant_bits"],
            md["skip_layers"], md["hook_point"],
        )
    return metadata


def find_hf_model(module: Any) -> Any:
    """Unwrap FSDP/compile wrappers down to the HF model that owns ``config``.

    ``install_turboquant_fake_quant`` needs head_dim / num_hidden_layers and a
    config object to flip ``_attn_implementation`` on, none of which an FSDP
    wrapper exposes.
    """
    seen = set()
    cur = module
    for _ in range(10):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        cfg = getattr(cur, "config", None)
        text_cfg = getattr(cfg, "text_config", cfg)
        if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
            return cur
        cur = (
            getattr(cur, "_fsdp_wrapped_module", None)
            or getattr(cur, "module", None)
            or getattr(cur, "_orig_mod", None)
        )
    return None


def ensure_actor_quantizer(actor_module: Any, config: Any = None):
    """Install the TurboQuant fake quantizer on the actor if the scheme asks for it.

    The repo has no working auto-activation for the KV plugin: the only
    candidate hook, ``envs/mopd-pylibs/sitecustomize.py``, only
    disables Ray tracing and never imports ``qat_kv``, so
    ``verl_plugin._attach_actor_model`` is never reached.  (That also explains
    why every historical run reported ``rollout_kv_verified=false``.)  Rather
    than depend on that machinery, the actor installs its own quantizer here.

    Returns the installed state, or ``None`` when no TurboQuant scheme is set.
    """
    if os.environ.get("QAT_STATEQUANT_TRAINING", "0") == "1":
        if os.environ.get("QAT_KV_SCHEME"):
            raise ValueError("StateQuant recurrent-state QAT requires ordinary attention KV to stay native")
        from opqad.integrations.statequant_training import (
            install_statequant_training, get_statequant_training_state,
        )
        from opqad.quantization.statequant_mamba2 import StateQuantMamba2Config
        existing = get_statequant_training_state(actor_module)
        if existing is not None:
            return existing
        model = find_hf_model(actor_module)
        return install_statequant_training(model, StateQuantMamba2Config(
            mode=os.environ.get("QAT_STATEQUANT_MODE", "pace4")))

    # Native NVFP4 KV has its own post-RoPE STE and codec. The generic scheme
    # resolver below only covers older/TurboQuant fake quantizers, so dispatch
    # native NVFP4 before resolving QAT_KV_SCHEME. Never stack a second
    # attention quantizer on top of the NVFP4 wrapper.
    if os.environ.get("QAT_NVFP4_KV_STE", "0").strip().lower() in {"1", "true", "yes", "on"}:
        from opqad.integrations.nvfp4_kv_ste import get_installed_nvfp4_state

        state = get_installed_nvfp4_state(actor_module)
        if state is None:
            raise RuntimeError(
                "QAT_NVFP4_KV_STE is enabled, but no native NVFP4 KV STE state "
                f"is installed under {type(actor_module).__name__}; refusing to "
                "fall through to the generic KV quantizer"
            )
        if not getattr(state, "enabled", False):
            raise RuntimeError("native NVFP4 KV STE state is present but disabled")

        metadata = {
            "plugin": "opqad.native_nvfp4_kv_ste",
            "worker_role": "actor",
            "scheme": "nvfp4",
            "status": "enabled_nvfp4",
            "actor_kv_status": "enabled_nvfp4",
            "actor_kv_fake_quantized": True,
            "actor_kv_quantized": True,
            "actor_kv_hook_applied": True,
            "kv_hook_applied": True,
            "projection_hook_applied": False,
            "hook_point": "post_rope_attention_kv",
            "format": "native_nvfp4_e2m1_block16_e4m3_scale",
            "num_layers": int(getattr(state, "num_layers", 0)),
            "quantized_layers": sorted(getattr(state, "layers", ()) or ()),
            "calls": int(getattr(state, "calls", 0)),
        }
        # Publish on the wrapper VeRL holds and the underlying model so
        # metadata readers can find it through either ownership path.
        setattr(actor_module, "_qat_kv_metadata", metadata)
        model = getattr(state, "model", None)
        if model is not None:
            setattr(model, "_qat_kv_metadata", metadata)
        LOGGER.info(
            "native NVFP4 KV STE already installed: layers=%d calls=%d hook=%s",
            metadata["num_layers"], metadata["calls"], metadata["hook_point"],
        )
        return state

    scheme = resolve_scheme(config)
    if not is_quantized_scheme(scheme):
        return None

    from opqad.integrations.turboquant_attn_hook import get_installed_state, install_turboquant_fake_quant

    existing = get_installed_state(actor_module)
    if existing is not None:
        return existing

    model = find_hf_model(actor_module)
    if model is None:
        raise RuntimeError(
            f"KV scheme {scheme!r} requested but no HF model with a usable config "
            f"was found under {type(actor_module).__name__}; cannot install the "
            "fake quantizer"
        )
    state = install_turboquant_fake_quant(model, scheme)
    # get_installed_state walks wrappers from the outside in; make sure the
    # lookup succeeds from the module dp_actor actually holds.
    if get_installed_state(actor_module) is None:
        setattr(actor_module, "_qat_turboquant_state", state)
    LOGGER.info(
        "%s: actor TurboQuant QAT installed via dp_actor: scheme=%s skip=%s",
        __name__, scheme, sorted(state.skip_layers),
    )
    return state
