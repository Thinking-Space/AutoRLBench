"""Weight/activation fake-quant for the QAT actor (NVFP4 via NVIDIA modelopt).

This is the W/A counterpart of :mod:`opqad.quantization.tq_scheme`, which handles the KV
cache.  The two are independent: an arm can run KV-only, W/A-only, or both.

What gets installed
-------------------
``mtq``'s ``TensorQuantizer`` modules wrap every ``nn.Linear``'s input and
weight.  Forward does quantize->dequantize in BF16 (fake-quant); backward is
modelopt's clipped straight-through estimator -- gradient passes unchanged
inside [-amax, amax] and is zeroed outside.  The tensors stay BF16 throughout,
so this costs time and saves nothing; it exists to make training see the
quantization error that NVFP4 serving will introduce.

Why the state comes from a file
-------------------------------
``NVFP4_DEFAULT_CFG`` declares ``algorithm: "max"``: the per-16-value block
scale is ``dynamic`` (computed at runtime, needs no data) but the per-tensor
amax is *calibrated*.  Re-calibrating at every job start would make each arm
see a slightly different amax and would burn several minutes per rank.  Instead
``tools/analysis/ptq_nvfp4.py`` writes ``modelopt_state.pt`` once, and every arm
restores the same frozen amax from it.

This module refuses to install without that file rather than silently
calibrating, because an uncalibrated amax is not a smaller version of the
right answer -- it is a different experiment.

Install point
-------------
``verl.workers.fsdp_workers._build_model_optimizer``, *before* the FSDP wrap:
``mtq`` replaces module attributes, which FSDP2's flat-parameter sharding does
not tolerate after the fact.  Only ``role == "actor"`` is quantized; the
reference/teacher model stays BF16, which is what makes the distillation
signal a full-precision target.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

LOGGER = logging.getLogger(__name__)

# modelopt config name per scheme.
SCHEMES = {
    "nvfp4": "NVFP4_DEFAULT_CFG",       # W4A4
    "nvfp4_w4a16": "W4A16_NVFP4_CFG",   # weights only
}


class WAQuantState:
    """Live handle on an installed W/A quantizer.

    Deliberately mirrors the surface ``TurboQuantAttnState`` exposes, because
    the block-alignment term in ``dp_actor`` consumes a quantizer state and
    should not care which axis is quantized: it aligns per-block hidden states
    between a quantized student and a full-precision reference, and NVFP4 W/A
    perturbs those hidden states exactly as KV quantization does.
    """

    def __init__(self, scheme: str, config_name: str, quantizers, state_path: str):
        self.scheme = scheme
        self.config_name = config_name
        self.quantizers = list(quantizers)
        self.state_path = state_path
        self.enabled = True

    @contextlib.contextmanager
    def disabled(self):
        """Run the same weights in full precision.

        Only used when block alignment recomputes its reference from the
        current weights.  The arms here set ``QAT_BLOCK_ALIGN_STATIC_REF=1``
        and pass a frozen model instead, so this is a correctness backstop
        rather than the hot path.
        """
        previously = [q.is_enabled for q in self.quantizers]
        was = self.enabled
        try:
            for q in self.quantizers:
                q.disable()
            self.enabled = False
            yield
        finally:
            for q, on in zip(self.quantizers, previously):
                if on:
                    q.enable()
            self.enabled = was

    def __repr__(self) -> str:
        return (
            f"WAQuantState(scheme={self.scheme!r}, config={self.config_name!r}, "
            f"quantizers={len(self.quantizers)}, enabled={self.enabled})"
        )


def get_installed_wa_state(module: Any):
    """Find the W/A state under ``module``, walking FSDP/compile wrappers."""
    seen = 0
    cur = module
    while cur is not None and seen < 8:
        state = getattr(cur, "_qat_wa_state", None)
        if isinstance(state, WAQuantState):
            return state
        cur = (
            getattr(cur, "_fsdp_wrapped_module", None)
            or getattr(cur, "module", None)
            or getattr(cur, "_orig_mod", None)
        )
        seen += 1
    return None


def resolve_wa_scheme(config: Any = None) -> str:
    """Return the configured W/A scheme name, or "" when disabled."""
    scheme = os.environ.get("QAT_WA_SCHEME", "").strip().lower()
    if not scheme and config is not None:
        scheme = str(getattr(config, "wa_scheme", "") or "").strip().lower()
    if scheme in ("", "none", "bf16", "off", "0"):
        return ""
    if scheme not in SCHEMES:
        raise ValueError(
            f"unknown QAT_WA_SCHEME {scheme!r}; expected one of {sorted(SCHEMES)}"
        )
    return scheme


def install_wa_fake_quant(model, scheme: str, state_path: str):
    """Restore the calibrated modelopt quantizer tree onto ``model`` in place.

    ``model`` must be the raw HF module, on its final device, before FSDP.
    """
    import torch
    import modelopt.torch.opt as mto
    from modelopt.torch.quantization.nn import TensorQuantizer

    if not state_path or not os.path.exists(state_path):
        raise FileNotFoundError(
            f"QAT_WA_SCHEME={scheme} needs a calibrated modelopt state, but "
            f"QAT_WA_STATE={state_path!r} does not exist.  Produce it with "
            f"tools/analysis/ptq_nvfp4.py --tier {'w4a4' if scheme == 'nvfp4' else 'w4a16'}."
        )

    blob = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "modelopt_state" not in blob:
        raise ValueError(
            f"{state_path} is not the two-part blob this expects "
            f"({{'modelopt_state': ..., 'quantizer_buffers': ...}}); it was "
            "probably written by a pre-mto.save version of ptq_nvfp4.py"
        )

    mto.restore_from_modelopt_state(model, modelopt_state=blob["modelopt_state"])

    buffers = blob.get("quantizer_buffers") or {}
    missing, unexpected = model.load_state_dict(buffers, strict=False)
    # `missing` lists every non-quantizer weight, which is expected here since
    # `buffers` only holds amax.  `unexpected` is not: it means the restored
    # tree does not have a quantizer the calibration produced.
    if unexpected:
        raise RuntimeError(
            f"modelopt state restore left {len(unexpected)} unmatched quantizer "
            f"buffers, e.g. {sorted(unexpected)[:5]}"
        )

    quantizers = [m for m in model.modules() if isinstance(m, TensorQuantizer)]
    n_calibrated = sum(1 for k in buffers if k.endswith("_amax"))
    LOGGER.info(
        "%s: actor W/A fake-quant installed: scheme=%s config=%s "
        "quantizers=%d calibrated_amax=%d state=%s",
        __name__, scheme, SCHEMES[scheme], len(quantizers), n_calibrated, state_path,
    )
    if not quantizers:
        raise RuntimeError(
            f"restoring {state_path} installed no TensorQuantizer; the model "
            "would train in plain BF16 while the arm claims to be quantized"
        )
    if n_calibrated == 0:
        raise RuntimeError(
            f"{state_path} carries no calibrated amax; NVFP4's per-tensor "
            "scale would be uninitialised"
        )
    return WAQuantState(scheme, SCHEMES[scheme], quantizers, state_path)


def ensure_actor_wa_quantizer(model, role: str = "actor", config: Any = None):
    """Install the W/A fake quantizer on the actor if an arm asks for it.

    Returns the installed state, or ``None`` when the scheme is off or the
    module is not the actor.
    """
    scheme = resolve_wa_scheme(config)
    if not scheme:
        return None
    if role != "actor":
        LOGGER.info(
            "%s: QAT_WA_SCHEME=%s set but role=%s; leaving this model in BF16",
            __name__, scheme, role,
        )
        return None
    if getattr(model, "_qat_wa_state", None) is not None:
        return model._qat_wa_state

    state = install_wa_fake_quant(model, scheme, os.environ.get("QAT_WA_STATE", ""))
    model._qat_wa_state = state
    return state
