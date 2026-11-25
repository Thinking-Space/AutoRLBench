"""Opt-in native Transformers Nemotron-H checkpoint loading.

NVIDIA's original checkpoint uses ``backbone`` while native Transformers uses
``model``. Shape validation reads safetensors headers and meta parameters only.
"""
import json
import os
from pathlib import Path


def native_nemotron_enabled():
    return os.environ.get("QAT_NEMOTRON_NATIVE_LOADER", "0") == "1"


def native_nemotron_config(path, **kwargs):
    from transformers import NemotronHConfig
    config = NemotronHConfig.from_pretrained(path, **kwargs)
    config.architectures = ["NemotronHForCausalLM"]
    config.auto_map = {}
    return config


def validate_native_checkpoint(path):
    import torch
    from safetensors import safe_open
    from transformers import NemotronHForCausalLM
    path = Path(path)
    config = native_nemotron_config(path)
    with torch.device("meta"):
        model = NemotronHForCausalLM(config)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    actual = {}
    original = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                mapped = "model." + name[len("backbone."):] if name.startswith("backbone.") else name
                if mapped in actual:
                    raise ValueError(f"duplicate mapped checkpoint key: {mapped}")
                actual[mapped] = tuple(handle.get_slice(name).get_shape())
                original[name] = mapped
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    mismatched = {k: {"model": expected[k], "checkpoint": actual[k]}
                  for k in expected.keys() & actual.keys() if expected[k] != actual[k]}
    if missing or unexpected or mismatched:
        raise ValueError(json.dumps(dict(missing=missing, unexpected=unexpected, shape_mismatch=mismatched)))
    return dict(status="passed", checkpoint=str(path), keys=len(actual),
                mapped_backbone_keys=sum(k.startswith("backbone.") for k in original),
                parameter_count=sum(v.numel() for v in model.parameters()),
                key_mapping={"^backbone\\.": "model."}, missing=[], unexpected=[], shape_mismatch={})


def load_native_nemotron(path, *, config=None, torch_dtype=None, **kwargs):
    from transformers import NemotronHForCausalLM
    config = config or native_nemotron_config(path)
    kwargs.pop("trust_remote_code", None)
    model, info = NemotronHForCausalLM.from_pretrained(
        path, config=config, torch_dtype=torch_dtype,
        key_mapping={r"^backbone\.": "model."}, output_loading_info=True, **kwargs)
    if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
        raise RuntimeError(f"native Nemotron checkpoint load incomplete: {info}")
    model._opqad_native_loading_info = info
    return model


def checkpoint_generation_config(config):
    """Normalize unused sampling flags in a checkpoint metadata copy.

    The original NVIDIA generation config predates Transformers' strict saving
    validation: it records top_p=0.95 while the implicit do_sample is false.
    Explicit rollout SamplingParams are independent of this metadata.
    """
    from copy import deepcopy
    from transformers import GenerationConfig
    saved = deepcopy(config)
    if not saved.do_sample:
        defaults = GenerationConfig()
        for name in ("temperature", "top_p", "top_k", "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff"):
            if hasattr(saved, name):
                setattr(saved, name, getattr(defaults, name))
    saved.validate(strict=True)
    return saved
