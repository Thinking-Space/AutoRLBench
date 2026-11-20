#!/usr/bin/env python3
"""从FSDP checkpoint加载模型并提取逐层kv_align数据"""

import sys
import torch
import json
from pathlib import Path
import argparse
from collections import OrderedDict

sys.path.insert(0, "/historical/home/QAT")
sys.path.insert(0, "/historical/home/QAT/vendor/verl")

def load_fsdp_checkpoint(ckpt_dir, device='cuda:0'):
    """加载FSDP checkpoint"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt_path = Path(ckpt_dir)
    base_model_path = "/historical/shared/qat/models/Nemotron-Research-Reasoning-Qwen-1.5B-brorl"

    print(f"加载基础模型架构: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # 检查checkpoint文件
    actor_dir = ckpt_path / "actor"
    rank0_model = actor_dir / "model_world_size_2_rank_0.pt"
    rank1_model = actor_dir / "model_world_size_2_rank_1.pt"

    if not rank0_model.exists() or not rank1_model.exists():
        print(f"警告: 找不到FSDP checkpoint文件，使用基础模型")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        return model.to(device), tokenizer

    print(f"加载FSDP checkpoint: {ckpt_path}")
    print(f"  Rank 0: {rank0_model}")
    print(f"  Rank 1: {rank1_model}")

    # 加载两个rank的state dict
    state_dict_rank0 = torch.load(rank0_model, map_location='cpu')
    state_dict_rank1 = torch.load(rank1_model, map_location='cpu')

    # 合并state dict（简化版：假设参数按rank切分）
    merged_state_dict = OrderedDict()

    # FSDP的state dict结构：
    # - 可能有 'model' 或 '_forward_module' 等前缀
    # - 参数可能被flat化

    # 尝试提取实际的模型参数
    if 'model' in state_dict_rank0:
        state_dict_rank0 = state_dict_rank0['model']
    if 'model' in state_dict_rank1:
        state_dict_rank1 = state_dict_rank1['model']

    print(f"Rank 0 keys: {len(state_dict_rank0)}")
    print(f"Rank 1 keys: {len(state_dict_rank1)}")

    # 简化方案：直接用rank0的参数（如果完整的话）
    # 完整的FSDP合并需要知道sharding策略
    print("尝试直接使用rank 0的参数...")

    # 移除可能的前缀
    cleaned_state_dict = OrderedDict()
    for key, value in state_dict_rank0.items():
        # 移除 _fsdp_wrapped_module. 等前缀
        clean_key = key
        for prefix in ['_fsdp_wrapped_module.', '_forward_module.', 'module.']:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        cleaned_state_dict[clean_key] = value

    # 尝试加载
    try:
        missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"加载完成: missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 0:
            print(f"  缺失的keys（前10个）: {list(missing)[:10]}")
    except Exception as e:
        print(f"警告: 无法加载checkpoint ({e})，使用基础模型")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    return model.to(device), tokenizer

def compute_layerwise_kv_align(model, tokenizer, kv_scheme, n_samples=64, device='cuda:0'):
    """计算逐层kv_align"""

    # 测试prompts
    test_prompts = [
        "Solve the equation: 2x + 5 = 13",
        "What is the derivative of x^2 + 3x?",
        "Explain the Pythagorean theorem.",
        "Calculate: 15 * 24 + 7",
        "What is the capital of France?",
        "Write a Python function to reverse a string.",
        "Explain Newton's second law of motion.",
        "What are the prime numbers less than 20?",
        "Simplify: (x+2)(x-3)",
        "What is photosynthesis?",
        "Find the area of a circle with radius 5.",
        "Explain the water cycle.",
        "What is recursion in programming?",
        "Solve: 3x - 7 = 2x + 5",
        "What causes seasons?",
        "Explain binary search algorithm.",
    ] * (n_samples // 16 + 1)

    test_prompts = test_prompts[:n_samples]

    print(f"\nTokenizing {n_samples} samples...")
    inputs = tokenizer(
        test_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    model.eval()

    print(f"前向推理...")
    layer_results = {'key': [], 'value': []}

    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        n_layers = len(past_key_values)

        print(f"\n计算逐层NMSE (共{n_layers}层)...")

        for layer_idx in range(n_layers):
            key_states, value_states = past_key_values[layer_idx]

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

            if (layer_idx + 1) % 5 == 0 or layer_idx == n_layers - 1:
                print(f"  Layer {layer_idx}: key_nmse={key_nmse:.6f}, value_nmse={value_nmse:.6f}")

    return layer_results

def apply_fake_quant(tensor, kv_scheme, kv_type):
    """应用fake quantization"""

    if kv_scheme == "turboquant_k3v4_nc":
        bits = 3 if kv_type == 'key' else 4
    elif kv_scheme == "turboquant_k2v3_nc":
        bits = 2 if kv_type == 'key' else 3
    elif kv_scheme == "int4_per_token_head":
        bits = 4
    else:
        raise ValueError(f"Unknown kv_scheme: {kv_scheme}")

    # Per-token-per-head uniform quantization
    # tensor: [B, H, T, D]
    x_min = tensor.min(dim=-1, keepdim=True)[0]
    x_max = tensor.max(dim=-1, keepdim=True)[0]

    n_levels = 2 ** bits
    scale = (x_max - x_min) / (n_levels - 1)
    scale = scale.clamp(min=1e-8)

    x_quant = ((tensor - x_min) / scale).round().clamp(0, n_levels - 1)
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
    parser.add_argument('--checkpoint', required=True, help='Checkpoint directory (e.g., .../global_step_30)')
    parser.add_argument('--kv-scheme', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-samples', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    print("=" * 70)
    print(f"从 Checkpoint 提取逐层 KV Align")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"KV Scheme: {args.kv_scheme}")
    print(f"Samples: {args.n_samples}")
    print(f"Device: {args.device}")

    # 加载模型
    model, tokenizer = load_fsdp_checkpoint(args.checkpoint, device=args.device)

    # 计算逐层kv_align
    results = compute_layerwise_kv_align(
        model, tokenizer, args.kv_scheme,
        n_samples=args.n_samples,
        device=args.device
    )

    # 保存结果
    output_data = {
        'checkpoint': args.checkpoint,
        'kv_scheme': args.kv_scheme,
        'n_samples': args.n_samples,
        'layer_results': results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"✓ 结果已保存: {output_path}")

    # 统计
    key_nmses = [r['nmse'] for r in results['key']]
    value_nmses = [r['nmse'] for r in results['value']]

    print(f"\nKey NMSE 统计:")
    print(f"  Min:  {min(key_nmses):.6f}")
    print(f"  Max:  {max(key_nmses):.6f}")
    print(f"  Mean: {sum(key_nmses)/len(key_nmses):.6f}")

    print(f"\nValue NMSE 统计:")
    print(f"  Min:  {min(value_nmses):.6f}")
    print(f"  Max:  {max(value_nmses):.6f}")
    print(f"  Mean: {sum(value_nmses)/len(value_nmses):.6f}")

if __name__ == "__main__":
    main()
