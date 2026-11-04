#!/usr/bin/env python3
"""Create a small PiSSA residual-base bundle for the AutoRLBench probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(2026)
    args.output.mkdir(parents=True, exist_ok=True)
    base_dir = args.output / "residual_base"
    adapter_dir = args.output / "pissa_adapter"

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        init_lora_weights="pissa_niter_2",
        bias="none",
    )
    model = get_peft_model(model, config)

    # PiSSA rewrites the wrapped base layers to W_res = W_pretrained - delta_init.
    # Save the adapter first, then remove its wrappers *without merging* so the
    # plain Transformers checkpoint contains the residual base weights.
    model.save_pretrained(adapter_dir, safe_serialization=True)
    residual_model = model.unload()
    residual_model.save_pretrained(base_dir, safe_serialization=True)
    tokenizer.save_pretrained(base_dir)

    manifest = {
        "schema_version": 1,
        "source_model": args.model,
        "source_revision": args.revision,
        "rank": args.rank,
        "target_modules": ["q_proj", "v_proj"],
        "init_lora_weights": "pissa_niter_2",
        "dtype": "bfloat16",
        "seed": 2026,
        "residual_base": str(base_dir),
        "adapter": str(adapter_dir),
    }
    (args.output / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
