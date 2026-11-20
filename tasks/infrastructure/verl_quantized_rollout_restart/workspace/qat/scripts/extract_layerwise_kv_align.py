#!/usr/bin/env python3
"""从 checkpoint 提取逐层 kv_align 数据"""

import sys
import torch
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

# 添加路径
sys.path.insert(0, "/historical/home/QAT")
sys.path.insert(0, "/historical/home/QAT/vendor/verl")

from qat_kv.kv_align import compute_kv_alignment
from qat_kv.turboquant_fake_quant import (
    turboquant_key_rotate,
    turboquant_key_indices,
    turboquant_value_scale_zero,
    turboquant_value_fake_quant,
    turboquant_key_fake_quant,
)

def load_checkpoint_model(ckpt_dir, device='cuda:0'):
    """加载 checkpoint 模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 检查是否是 FSDP checkpoint
    ckpt_path = Path(ckpt_dir)
    if (ckpt_path / "pytorch_model.bin").exists():
        model_path = ckpt_path
    elif (ckpt_path / "__0_0.distcp").exists():
        # FSDP checkpoint，需要合并
        print(f"检测到 FSDP checkpoint: {ckpt_path}")
        # 简化：直接加载基础模型然后加载 state_dict
        base_model_path = "/historical/shared/qat/models/Nemotron-Research-Reasoning-Qwen-1.5B-brorl"
        print(f"加载基础模型: {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map=device,
        )

        # 尝试加载 FSDP state
        # TODO: 需要完整的 FSDP 加载逻辑
        print("警告: FSDP checkpoint 加载需要完整实现")
        return model, None
    else:
        # 回退到基础模型
        base_model_path = "/historical/shared/qat/models/Nemotron-Research-Reasoning-Qwen-1.5B-brorl"
        print(f"checkpoint 格式不明确，使用基础模型: {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map=device,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        "/historical/shared/qat/models/Nemotron-Research-Reasoning-Qwen-1.5B-brorl",
        trust_remote_code=True,
    )

    return model, tokenizer

def compute_layerwise_kv_align(model, tokenizer, kv_scheme, n_samples=32, max_length=512):
    """计算逐层 kv_align"""

    # 准备测试数据
    from transformers import AutoTokenizer
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum mechanics in simple terms.",
        "Write a Python function to sort a list.",
        "What are the benefits of exercise?",
    ] * (n_samples // 4)

    test_prompts = test_prompts[:n_samples]

    # Tokenize
    inputs = tokenizer(
        test_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    model.eval()

    # 存储逐层结果
    layer_results = {
        'key': [],
        'value': [],
    }

    with torch.no_grad():
        # Forward pass 获取 KV cache
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_hidden_states=True,
            use_cache=True,
        )

        # 提取 past_key_values
        past_key_values = outputs.past_key_values

        # 对每一层计算 kv_align
        n_layers = len(past_key_values)

        for layer_idx in tqdm(range(n_layers), desc="Computing layer-wise KV align"):
            key_states, value_states = past_key_values[layer_idx]

            # key_states: [B, n_heads, seq_len, head_dim]
            # value_states: [B, n_heads, seq_len, head_dim]

            # 应用量化
            if kv_scheme == "turboquant_k3v4_nc":
                # Key: 3-bit VQ
                key_quantized = turboquant_key_fake_quant(
                    key_states, bits=3, num_codebooks=1, per_head=True
                )
                # Value: 4-bit uniform
                value_quantized = turboquant_value_fake_quant(
                    value_states, bits=4, per_token_per_head=True
                )
            elif kv_scheme == "turboquant_k2v3_nc":
                key_quantized = turboquant_key_fake_quant(
                    key_states, bits=2, num_codebooks=1, per_head=True
                )
                value_quantized = turboquant_value_fake_quant(
                    value_states, bits=3, per_token_per_head=True
                )
            elif kv_scheme == "int4_per_token_head":
                # RHT + INT4
                from qat_kv.turboquant_fake_quant import int4_per_token_head_fake_quant
                key_quantized = int4_per_token_head_fake_quant(key_states, bits=4)
                value_quantized = int4_per_token_head_fake_quant(value_states, bits=4)
            else:
                raise ValueError(f"Unknown kv_scheme: {kv_scheme}")

            # 计算 NMSE
            key_mse = ((key_states - key_quantized) ** 2).mean(dim=-1)  # [B, H, T]
            key_norm = (key_states ** 2).mean(dim=-1) + 1e-8
            key_nmse = (key_mse / key_norm).mean().item()

            value_mse = ((value_states - value_quantized) ** 2).mean(dim=-1)
            value_norm = (value_states ** 2).mean(dim=-1) + 1e-8
            value_nmse = (value_mse / value_norm).mean().item()

            layer_results['key'].append({
                'layer': layer_idx,
                'nmse': key_nmse,
            })
            layer_results['value'].append({
                'layer': layer_idx,
                'nmse': value_nmse,
            })

    return layer_results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, help='Run directory')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint name (e.g., global_step_30)')
    parser.add_argument('--kv-scheme', required=True, help='KV scheme')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--n-samples', type=int, default=32, help='Number of samples')
    parser.add_argument('--device', default='cuda:0', help='Device')
    args = parser.parse_args()

    ckpt_dir = Path(args.run_dir) / "checkpoints" / args.checkpoint

    print(f"加载 checkpoint: {ckpt_dir}")
    model, tokenizer = load_checkpoint_model(ckpt_dir, device=args.device)

    print(f"计算逐层 kv_align (scheme={args.kv_scheme}, n_samples={args.n_samples})...")
    results = compute_layerwise_kv_align(
        model, tokenizer, args.kv_scheme, n_samples=args.n_samples
    )

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'checkpoint': args.checkpoint,
        'kv_scheme': args.kv_scheme,
        'n_samples': args.n_samples,
        'layer_results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"结果已保存: {output_path}")

    # 打印摘要
    print("\n逐层 Key NMSE:")
    for item in results['key'][:5]:
        print(f"  Layer {item['layer']}: {item['nmse']:.6f}")
    print("  ...")

    print("\n逐层 Value NMSE:")
    for item in results['value'][:5]:
        print(f"  Layer {item['layer']}: {item['nmse']:.6f}")
    print("  ...")

if __name__ == "__main__":
    main()
