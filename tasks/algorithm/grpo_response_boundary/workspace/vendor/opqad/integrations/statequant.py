"""Thin integration of the original StateQuant Mamba2 vLLM plugin."""
from __future__ import annotations

import importlib.metadata
import math
import os

UPSTREAM_REVISION = "71d6db63fa5d330dc21c296764c133f47ca20929"
FORMATS = ("stock-fp32", "stock-bf16", "pace4", "int4v", "bf16")


def configure_environment(mode: str, *, window: int = 8, buffer_bits: int = 8) -> dict[str, str]:
    """Select only this engine's plugin before vLLM is imported."""
    if mode not in FORMATS:
        raise ValueError(f"Unknown StateQuant mode: {mode}")
    if window < 1 or window > 16 or buffer_bits not in (8, 16):
        raise ValueError("Use a window in [1, 16] and an 8/16-bit buffer")
    for name in tuple(os.environ):
        if name.startswith("STATEQUANT_VLLM"):
            del os.environ[name]
    env = {
        "VLLM_PLUGINS": "opqad_statequant",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if not mode.startswith("stock-"):
        env.update({
            "STATEQUANT_VLLM": mode,
            "STATEQUANT_VLLM_M": str(window),
            "STATEQUANT_VLLM_BUFFER_BITS": str(buffer_bits),
            "STATEQUANT_VLLM_LAYERS": "mamba2",
        })
    os.environ.update(env)
    return env


def register() -> None:
    """vLLM entry point: retain StateQuant's kernels, layout and patch order."""
    if not os.environ.get("STATEQUANT_VLLM"):
        return
    version = importlib.metadata.version("vllm")
    if not version.startswith("0.32."):
        raise RuntimeError(f"This integration pins vLLM 0.32; found {version}")
    from statequant.integrations.vllm_plugin import register as upstream_register

    upstream_register()


def worker_cache_report(worker) -> dict:
    """Inspect the actual worker tensors, outside the measured decode path."""
    model = worker.model_runner.model
    layers = []
    for name, layer in model.named_modules():
        if type(layer).__name__ != "MambaMixer2":
            continue
        state = layer.kv_cache[1]
        layout = getattr(layer, "_sq_layout", None)
        layers.append({
            "name": name,
            "state_dtype": str(state.dtype),
            "state_shape": list(state.shape),
            "state_bytes_per_slot": math.prod(state.shape[1:]) * state.element_size(),
            "packed_layout": str(layout) if layout is not None else None,
            "layout_bytes_per_slot": layout.nbytes if layout is not None else None,
            "head_dim": layer.head_dim,
            "state_dim": layer.ssm_state_size,
            "heads": layer.num_heads,
            "groups": layer.n_groups,
        })
    config = worker.vllm_config.cache_config
    return {
        "layers": layers,
        "enable_prefix_caching": config.enable_prefix_caching,
        "mamba_cache_mode": config.mamba_cache_mode,
        "use_replayssm": config.use_replayssm,
        "statequant_version": importlib.metadata.version("statequant"),
    }


def verify_cache_reports(reports: list[dict], mode: str, expected_layers: int) -> None:
    """Require the expected allocation; an enabled flag alone is insufficient."""
    if not reports:
        raise RuntimeError("No worker cache report returned")
    for report in reports:
        if len(report["layers"]) != expected_layers:
            raise RuntimeError(f"Expected {expected_layers} Mamba2 layers; got {len(report['layers'])}")
        if report["enable_prefix_caching"] or report["use_replayssm"]:
            raise RuntimeError("This comparison disables prefix caching and ReplaySSM")
        for layer in report["layers"]:
            if mode.startswith("stock-"):
                expected = "torch.float32" if mode == "stock-fp32" else "torch.bfloat16"
                if layer["packed_layout"] is not None or layer["state_dtype"] != expected:
                    raise RuntimeError(f"Unexpected stock state: {layer}")
            elif (layer["state_dtype"] != "torch.uint8"
                  or layer["layout_bytes_per_slot"] != layer["state_bytes_per_slot"]):
                raise RuntimeError(f"Packed state allocation was not installed: {layer}")


class StateQuantWorkerExtension:
    """Use vLLM's native named RPC interface; no function-pickle fallback."""

    def statequant_cache_report(self) -> dict:
        return worker_cache_report(self)
