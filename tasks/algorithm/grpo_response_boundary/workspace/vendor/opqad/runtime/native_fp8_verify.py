"""Inspect native FP8 KV dtype, layer scales and cache tensors inside vLLM."""
from __future__ import annotations
import math

def worker_fp8_evidence(worker):
    from opqad.runtime.native_tq_verify import _attention_layers, _layers_from_model, _layer_index
    holders = [worker, getattr(worker, "worker", None)]
    holders = [h for h in holders if h is not None]
    runner = next((h.model_runner for h in holders if getattr(h, "model_runner", None) is not None), None)
    cfg = next((h.vllm_config for h in holders if getattr(h, "vllm_config", None) is not None), None)
    layers = _attention_layers(cfg) or _layers_from_model(getattr(runner, "model", None))
    rows = []
    for name, layer in sorted(layers.items()):
        idx = _layer_index(name)
        if idx is None: continue
        impl = getattr(layer, "impl", None)
        row = {"layer": idx, "kv_cache_dtype": str(getattr(layer, "kv_cache_dtype", None)),
               "backend": type(impl).__module__ + "." + type(impl).__qualname__,
               "query_quantizer": type(getattr(layer, "query_quant", None)).__qualname__}
        for kind in ("q", "k", "v"):
            value = getattr(layer, "_" + kind + "_scale_float", None)
            if value is None: value = getattr(layer, "_" + kind + "_scale", None)
            row[kind + "_scale"] = float(value) if value is not None else None
        rows.append(row)
    tensors = [{"dtype": str(t.dtype), "shape": list(t.shape), "device": str(t.device), "element_size": t.element_size()}
               for t in getattr(runner, "kv_caches", []) if hasattr(t, "dtype")]
    cache = getattr(cfg, "cache_config", None)
    return {"cache_dtype": str(getattr(cache, "cache_dtype", None)),
            "calculate_kv_scales": bool(getattr(cache, "calculate_kv_scales", False)),
            "layers": rows, "cache_tensors": tensors}

def verify_native_fp8_rollout(engine, num_layers):
    records = engine.collective_rpc(worker_fp8_evidence)
    failures = []
    for rank, record in enumerate(records):
        if record["cache_dtype"] not in ("fp8", "fp8_e4m3"): failures.append(f"rank{rank}: cache dtype")
        if not record["calculate_kv_scales"]: failures.append(f"rank{rank}: runtime scale calculation disabled")
        if {x["layer"] for x in record["layers"]} != set(range(num_layers)): failures.append(f"rank{rank}: layer coverage")
        for layer in record["layers"]:
            if layer["kv_cache_dtype"] not in ("fp8", "fp8_e4m3"): failures.append(f"rank{rank}: layer dtype")
            if not all(layer[k] is not None and math.isfinite(layer[k]) and layer[k] > 0 for k in ("k_scale", "v_scale")):
                failures.append(f"rank{rank}: invalid KV scales")
        if not record["cache_tensors"] or not all(t["element_size"] == 1 and t["dtype"] in ("torch.uint8", "torch.float8_e4m3fn") for t in record["cache_tensors"]):
            failures.append(f"rank{rank}: cache storage")
    if not records: failures.append("no worker evidence")
    if failures: raise RuntimeError("native FP8 audit failed: " + "; ".join(failures))
    return {"rollout_kv_verified": True, "ranks": records}
