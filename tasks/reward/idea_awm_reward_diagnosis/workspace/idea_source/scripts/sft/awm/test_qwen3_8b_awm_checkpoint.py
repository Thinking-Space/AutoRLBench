#!/usr/bin/env python3
"""Smoke-test a VERL/FSDP Hugging Face checkpoint with real SFT examples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        default="/local/user/models/qwen3_8b_awm_skill_sft_verl",
    )
    parser.add_argument("--step", type=int, default=1608)
    parser.add_argument(
        "--data",
        default="/workspace/idea_source/outputs/awm_qwen38_sft_data/sft_episode_skill_val.parquet",
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def extract_json(text: str) -> dict | None:
    text = text.strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def check_checkpoint(path: Path) -> dict:
    required = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
    missing = [name for name in required if not (path / name).is_file()]
    result = {"checkpoint": str(path), "missing": missing, "weight_bytes": 0, "tensor_count": 0}
    if missing:
        return result
    weight_path = path / "model.safetensors"
    result["weight_bytes"] = weight_path.stat().st_size
    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        result["tensor_count"] = len(keys)
        result["first_tensor"] = keys[0]
        result["first_shape"] = list(handle.get_tensor(keys[0]).shape)
    return result


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint = checkpoint_root / f"global_step_{args.step}"
    report: dict = {"checkpoint_check": check_checkpoint(checkpoint), "samples": []}
    if args.check_all:
        checkpoints = sorted(
            checkpoint_root.glob("global_step_*"), key=lambda item: int(item.name.removeprefix("global_step_"))
        )
        report["all_checkpoint_checks"] = [check_checkpoint(item) for item in checkpoints]
    if report["checkpoint_check"]["missing"]:
        raise SystemExit(json.dumps(report, indent=2))

    import pandas as pd

    rows = pd.read_parquet(args.data).head(args.num_samples).to_dict("records")
    print(f"Loading {checkpoint}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **load_kwargs)
    model.eval()

    input_device = next(model.parameters()).device
    for index, row in enumerate(rows):
        prompt = str(row["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        print(
            f"Generating sample {index + 1}/{len(rows)} ({inputs['input_ids'].shape[1]} input tokens)...",
            flush=True,
        )
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        parsed = extract_json(completion)
        report["samples"].append(
            {
                "index": index,
                "task_id": row.get("task_id"),
                "output": completion,
                "output_token_ids": generated[0, inputs["input_ids"].shape[1] :].detach().cpu().tolist(),
                "json_valid": parsed is not None,
                "json_keys": sorted(parsed) if parsed else [],
            }
        )

    report["summary"] = {
        "device": str(input_device),
        "samples": len(report["samples"]),
        "json_valid": sum(item["json_valid"] for item in report["samples"]),
        "expected_keys": ["episode_summary", "episode_skill", "step_skills"],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
