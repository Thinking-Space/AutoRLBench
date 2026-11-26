#!/usr/bin/env python3
"""从checkpoint提取逐层kv_align数据（简化版，直接用基础模型做fake quant前向）"""

import sys
import torch
import json
from pathlib import Path
import argparse

sys.path.insert(0, "/historical/home/QAT")

def compute_layerwise_kv_align_simple(kv_scheme, n_samples=64, device='cuda:0'):
    """使用基础模型计算逐层kv_align（不需要加载checkpoint）"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载基础模型...")
    base_model_path = "/historical/shared/qat/models/Nemotron-Research-Reasoning-Qwen-1.5B-brorl"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    # 测试数据
    test_prompts = [
        "Solve the equation: 2x + 5 = 13",
        "What is the derivative of x^2 + 3x?",
        "Explain the Pythagorean theorem.",
        "Calculate: 15 * 24 + 7",
        "What is the capital of France?",
        "Write a Python function to reverse a string.",
        "Explain Newton's second law of motion.",
        "What are the prime numbers less than 20?",
    ] * (n_samples // 8)

    test_prompts = test_prompts[:n_samples]

    print(f"Tokenizing {n_samples} samples...")
    inputs = tokenizer(
        test_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    model.eval()

    print(f"前向推理并计算逐层 NMSE...")
    layer_results = {'key': [], 'value': []}

    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        n_layers = len(past_key_values)

        print(f"模型有 {n_layers} 层")

        for layer_idx in range(n_layers):
            key_states, value_states = past_key_values[layer_idx]

            # key_states: [B, n_heads, seq_len, head_dim]
            # value_states: [B, n_heads, seq_len, head_dim]

            # 应用fake quantization
            key_quantized = apply_fake_quant(key_states, kv_scheme, 'key')
            value_quantized = apply_fake_quant(value_states, kv_scheme, 'value')

            # 计算NMSE
            key_nmse = compute_nmse(key_states, key_quantized)
            value_nmse = compute_nmse(value_states, value_quantized)

            layer_results['key'].append({
                'layer': layer_idx,
                'nmse': float(key_nmse),
            })
            layer_results['value'].append({
                'layer': layer_idx,
                'nmse': float(value_nmse),
            })

            if (layer_idx + 1) % 5 == 0:
                print(f"  Layer {layer_idx}: key_nmse={key_nmse:.6f}, value_nmse={value_nmse:.6f}")

    return layer_results

def apply_fake_quant(tensor, kv_scheme, kv_type):
    """应用fake quantization"""

    if kv_scheme == "turboquant_k3v4_nc":
        if kv_type == 'key':
            return turboquant_key_fake_quant_simple(tensor, bits=3)
        else:
            return uniform_fake_quant(tensor, bits=4)

    elif kv_scheme == "turboquant_k2v3_nc":
        if kv_type == 'key':
            return turboquant_key_fake_quant_simple(tensor, bits=2)
        else:
            return uniform_fake_quant(tensor, bits=3)

    elif kv_scheme == "int4_per_token_head":
        # INT4 with rotate-half
        return uniform_fake_quant(tensor, bits=4)

    else:
        raise ValueError(f"Unknown kv_scheme: {kv_scheme}")

def turboquant_key_fake_quant_simple(x, bits=3):
    """简化版TurboQuant key量化（不用真VQ，只做范围量化）"""
    # [B, H, T, D]

    # 简化：per-token-per-head的uniform量化
    B, H, T, D = x.shape

    # Flatten to [B*H*T, D]
    x_flat = x.reshape(-1, D)

    # Per-vector min/max
    x_min = x_flat.min(dim=-1, keepdim=True)[0]
    x_max = x_flat.max(dim=-1, keepdim=True)[0]

    # Quantize
    n_levels = 2 ** bits
    scale = (x_max - x_min) / (n_levels - 1)
    scale = scale.clamp(min=1e-8)

    x_quant = ((x_flat - x_min) / scale).round().clamp(0, n_levels - 1)
    x_dequant = x_quant * scale + x_min

    return x_dequant.reshape(B, H, T, D)

def uniform_fake_quant(x, bits=4):
    """Per-token-per-head uniform量化"""
    # [B, H, T, D]

    # Per (B, H, T) 的 min/max
    x_min = x.min(dim=-1, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0]

    n_levels = 2 ** bits
    scale = (x_max - x_min) / (n_levels - 1)
    scale = scale.clamp(min=1e-8)

    x_quant = ((x - x_min) / scale).round().clamp(0, n_levels - 1)
    x_dequant = x_quant * scale + x_min

    return x_dequant

def compute_nmse(original, quantized):
    """计算NMSE"""
    mse = ((original - quantized) ** 2).mean(dim=-1)  # [B, H, T]
    norm = (original ** 2).mean(dim=-1) + 1e-8
    nmse = (mse / norm).mean()
    return nmse.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kv-scheme', required=True, choices=['turboquant_k3v4_nc', 'turboquant_k2v3_nc', 'int4_per_token_head'])
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--n-samples', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    print("=" * 60)
    print(f"提取逐层 KV Align: {args.kv_scheme}")
    print("=" * 60)

    results = compute_layerwise_kv_align_simple(
        args.kv_scheme,
        n_samples=args.n_samples,
        device=args.device,
    )

    output_data = {
        'kv_scheme': args.kv_scheme,
        'n_samples': args.n_samples,
        'note': 'Computed using base model (not from checkpoint)',
        'layer_results': results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ 结果已保存: {output_path}")

    # 统计
    key_nmses = [r['nmse'] for r in results['key']]
    value_nmses = [r['nmse'] for r in results['value']]

    print(f"\nKey NMSE: min={min(key_nmses):.6f}, max={max(key_nmses):.6f}, mean={sum(key_nmses)/len(key_nmses):.6f}")
    print(f"Value NMSE: min={min(value_nmses):.6f}, max={max(value_nmses):.6f}, mean={sum(value_nmses)/len(value_nmses):.6f}")

if __name__ == "__main__":
    main()
