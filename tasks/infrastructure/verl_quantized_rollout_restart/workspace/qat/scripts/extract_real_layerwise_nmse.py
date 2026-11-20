#!/usr/bin/env python3
"""从真实FSDP训练checkpoint提取逐层KV量化NMSE（forward inference + fake quant）"""

import sys
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, "/historical/home/QAT")
from scripts.merge_fsdp_ckpt import load_model_from_fsdp_ckpt

BASE_MODEL = "/historical/shared/qat/models/Nemotron-Research-Reasoning-Qwen-1.5B-brorl"

TEST_PROMPTS = [
    "Solve the equation: 2x + 5 = 13",
    "What is the derivative of x^2 + 3x?",
    "Explain the Pythagorean theorem.",
    "Calculate: 15 * 24 + 7",
    "A train travels 60 miles in 1.5 hours. What is its average speed?",
    "Write a Python function to reverse a string.",
    "Explain Newton's second law of motion.",
    "What are the prime numbers less than 30?",
    "Simplify: (x+2)(x-3)",
    "If a rectangle has length 8 and width 5, find its area and perimeter.",
    "Find the area of a circle with radius 5.",
    "Explain how binary search works.",
    "What is the sum of the first 10 positive integers?",
    "Solve: 3x - 7 = 2x + 5",
    "A store sells apples at $0.50 each. How much do 24 apples cost?",
    "Explain recursion with a simple example.",
]


def bits_for_scheme(kv_scheme):
    if kv_scheme == "turboquant_k3v4_nc":
        return {"key": 3, "value": 4}
    if kv_scheme == "turboquant_k2v3_nc":
        return {"key": 2, "value": 3}
    if kv_scheme == "int4_per_token_head":
        return {"key": 4, "value": 4}
    raise ValueError(f"未知kv_scheme: {kv_scheme}")


def per_token_head_fake_quant(x, bits):
    # x: [B, H, T, D], 逐(token,head)向量的min-max均匀量化
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    n_levels = 2 ** bits
    scale = ((x_max - x_min) / (n_levels - 1)).clamp_min(1e-8)
    x_q = ((x - x_min) / scale).round().clamp(0, n_levels - 1)
    return x_q * scale + x_min


def compute_nmse(orig, quant):
    mse = ((orig - quant) ** 2).mean(dim=-1)
    norm = (orig ** 2).mean(dim=-1) + 1e-8
    return (mse / norm).mean().item()


def extract_layerwise(model, tokenizer, kv_scheme, device):
    bits = bits_for_scheme(kv_scheme)
    inputs = tokenizer(
        TEST_PROMPTS, return_tensors="pt", padding=True, truncation=True, max_length=256,
    ).to(device)

    model.eval()
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=True)

    cache = out.past_key_values
    n_layers = len(cache.layers)

    key_nmse, value_nmse = [], []
    for i in range(n_layers):
        layer = cache.layers[i]
        k, v = layer.keys, layer.values  # [B, H, T, D]
        k_q = per_token_head_fake_quant(k, bits["key"])
        v_q = per_token_head_fake_quant(v, bits["value"])
        key_nmse.append(compute_nmse(k, k_q))
        value_nmse.append(compute_nmse(v, v_q))

    return {"key_nmse": key_nmse, "value_nmse": value_nmse, "n_layers": n_layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-actor-dir", required=True)
    ap.add_argument("--kv-scheme", required=True, choices=["turboquant_k3v4_nc", "turboquant_k2v3_nc", "int4_per_token_head"])
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"加载checkpoint: {args.ckpt_actor_dir}")
    model = load_model_from_fsdp_ckpt(args.ckpt_actor_dir, BASE_MODEL, device=args.device)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"提取逐层NMSE: scheme={args.kv_scheme}, step={args.step}")
    result = extract_layerwise(model, tokenizer, args.kv_scheme, args.device)
    result["kv_scheme"] = args.kv_scheme
    result["step"] = args.step
    result["ckpt"] = args.ckpt_actor_dir

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"已保存: {out_path}")
    print(f"Key NMSE mean={sum(result['key_nmse'])/len(result['key_nmse']):.4f}")
    print(f"Value NMSE mean={sum(result['value_nmse'])/len(result['value_nmse']):.4f}")


if __name__ == "__main__":
    main()
