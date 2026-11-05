"""Weight loading and synchronization telemetry for the rollout integration."""
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open


def checkpoint_weights(model_path):
    files = sorted(Path(model_path).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint in {model_path}")
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def fingerprint(model):
    return {n: p.detach().reshape(-1)[:32].float().cpu().clone()
            for n, p in model.named_parameters()}


def finalize_weights(model, model_path):
    before = fingerprint(model)
    model.load_weights(checkpoint_weights(model_path))
    after = fingerprint(model)
    differences = [float((before[k] - after[k]).abs().max()) for k in before]
    record = {"sampled_parameters": len(before),
              "sampled_elements": sum(x.numel() for x in before.values()),
              "after_sync_max_abs_delta": max(differences, default=0.0),
              "changed_parameters": sum(x > 0 for x in differences)}
    trace = Path(os.environ.get("CASE_SYNC_TRACE", "outputs/sync_trace.jsonl"))
    trace.parent.mkdir(parents=True, exist_ok=True)
    with trace.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record
