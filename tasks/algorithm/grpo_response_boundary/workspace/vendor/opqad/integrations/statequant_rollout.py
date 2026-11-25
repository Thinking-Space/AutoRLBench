"""Formal StateQuant integration using the native vLLM Nemotron loader."""
import json
import os
from pathlib import Path
import time
import torch
from opqad.integrations.statequant import worker_cache_report, verify_cache_reports


def checkpoint_name(name):
    return "backbone." + name[6:] if name.startswith("model.") else name



def target_view(model, params, native, source, all_sources):
    """Locate a TP1 verification view with the installed author's mapper.

    Actual loading remains in NemotronHForCausalLM.load_weights. The mapper
    handles embeddings/embed_tokens and A_log/A in addition to packed QKV.
    """
    name = model.hf_to_vllm_mapper.map_name(checkpoint_name(native))
    if name is None:
        raise ValueError(f"Nemotron source unexpectedly skipped: {native}")
    target = params[name]
    for part in ("q", "k", "v"):
        marker = f".{part}_proj."
        if marker in native:
            prefix, suffix = native.split(marker)
            order = ("q", "k", "v")
            offset = sum(all_sources[prefix + f".{p}_proj." + suffix].shape[0]
                         for p in order[:order.index(part)])
            target = target.narrow(0, offset, source.shape[0])
            break
    # The official composed loader first copies A_log into the device FP32
    # parameter, then applies exp there. CPU exp has different final bits.
    expected = (-torch.exp(source.to(device=target.device, dtype=torch.float32))
                if native.endswith(".A_log") else source)
    if tuple(target.shape) != tuple(expected.shape):
        raise ValueError(f"Nemotron target shape mismatch: {native} -> {name}: "
                         f"{tuple(target.shape)} != {tuple(expected.shape)}")
    return target, expected

def _record(kind, report):
    print("OPQAD_STATEQUANT_" + kind.upper() + " " + json.dumps(report), flush=True)
    prefix = os.environ.get("OPQAD_STATEQUANT_SYNC_RECORD")
    if prefix:
        path = Path(prefix + "_rank_" + os.environ.get("RANK", "0") + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(dict(kind=kind, **report)) + "\n")


def verify_engine(engine):
    worker = engine.llm_engine.model_executor.driver_worker.worker
    report = worker_cache_report(worker)
    verify_cache_reports([report], "pace4", 21)
    impl = {type(layer).conv_ssm_forward.__module__ for layer in worker.model_runner.model.modules()
            if type(layer).__name__ == "MambaMixer2"}
    if impl != {"statequant.integrations.vllm_mamba2"}:
        raise RuntimeError(f"StateQuant author Mamba2 implementation missing: {impl}")
    report.update(implementations=sorted(impl), status="passed")
    _record("cache", report)
    return report


def load_and_verify_weights(model, weights):
    """Load all native tensors and check every target after QKV packing/A conversion."""
    start = time.monotonic()
    sources = dict(weights)
    if len(sources) != 263 or not all(name.startswith(("model.", "lm_head.")) for name in sources):
        raise RuntimeError(f"Expected all 263 native Nemotron weights, found {len(sources)}")
    loaded = model.load_weights((checkpoint_name(name), value) for name, value in sources.items())
    params = dict(model.named_parameters())
    mismatches = []
    for native, source in sources.items():
        target, expected = target_view(model, params, native, source, sources)
        if not torch.equal(target, expected.to(device=target.device, dtype=target.dtype)):
            mismatches.append(native)
    if mismatches:
        raise RuntimeError(f"Nemotron actor-to-rollout mismatch: {mismatches}")
    count = getattr(model, "_opqad_statequant_sync_count", 0) + 1
    model._opqad_statequant_sync_count = count
    _record("weight_sync", {"status": "passed", "sync_count": count,
        "source_weight_count": len(sources), "exactly_verified_sources": len(sources),
        "mamba_input_projections": sum("mixer.in_proj.weight" in name for name in sources),
        "native_vllm_loader": True, "seconds": time.monotonic()-start})
    return loaded
