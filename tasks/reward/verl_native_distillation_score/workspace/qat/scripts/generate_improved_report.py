#!/usr/bin/env python3
"""生成改进版的训练报告，解决 settings 不清晰和绘图质量差的问题"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from collections import defaultdict

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_metrics(json_path):
    """加载提取的训练指标"""
    with open(json_path, 'r') as f:
        return json.load(f)

def load_settings(run_dir):
    """加载运行配置"""
    settings_path = Path(run_dir) / "effective_settings.json"
    with open(settings_path, 'r') as f:
        return json.load(f)

def format_settings_summary(settings):
    """格式化配置摘要为清晰的字典"""
    key_settings = {
        "模型与数据": {
            "KV 量化方案": settings.get("kv_scheme", "N/A"),
            "模型": "Qwen-1.5B-brorl",
            "批大小": settings.get("global_batch", "N/A"),
            "学习率": settings.get("actor_lr", "N/A"),
            "训练步数": settings.get("updates", "N/A"),
        },
        "损失函数": {
            "block_align 系数": settings.get("block_align_coef", "N/A"),
            "block_align 指标": settings.get("block_align_metric", "N/A"),
            "kv_align 系数": settings.get("kv_align_coef", "N/A"),
            "OPD top-k": settings.get("reward_top_k", "N/A"),
            "目标函数": settings.get("objective", "N/A"),
        },
        "训练策略": {
            "Actor 模型精度": settings.get("actor_model_dtype", "N/A"),
            "FSDP 参数精度": settings.get("actor_fsdp_param_dtype", "N/A"),
            "FSDP reduce 精度": settings.get("actor_fsdp_reduce_dtype", "N/A"),
            "梯度检查点": settings.get("actor_gradient_checkpointing", "N/A"),
            "注意力实现": settings.get("actor_attention", "N/A"),
        },
        "Rollout 配置": {
            "量化 rollout": settings.get("rollout_kv_quantized", "N/A"),
            "Native TurboQuant": settings.get("native_turboquant_rollout", "N/A"),
            "Teacher KV dtype": settings.get("teacher_kv_cache_dtype", "N/A"),
            "Rollout KV 验证": settings.get("rollout_kv_verified", "N/A"),
        }
    }
    return key_settings

def create_settings_markdown(runs_info):
    """创建清晰的 settings 文档"""
    lines = ["# 当前训练配置总览\n"]
    lines.append(f"**训练任务**: 1.5B 模型三臂 KV 量化 QAT\n")
    lines.append(f"**运行时间**: 2026-09-08\n\n")

    for scheme, jobid, run_dir, _ in runs_info:
        lines.append(f"## {scheme} (Job {jobid})\n")

        settings = load_settings(run_dir)
        summary = format_settings_summary(settings)

        for section, params in summary.items():
            lines.append(f"### {section}\n")
            for key, value in params.items():
                lines.append(f"- **{key}**: `{value}`\n")
            lines.append("\n")

    return "\n".join(lines)

def plot_training_curves(metrics_data, output_path):
    """创建高质量的训练曲线图"""

    # 关键指标定义
    key_metrics = [
        ("actor/policy_loss", "Policy Loss", "loss"),
        ("actor/reward", "奖励 (Reward)", "reward"),
        ("val-topk/overlap_ratio", "Top-K 重叠率", "overlap"),
        ("block_align", "Block Align (NMSE)", "align"),
        ("actor/entropy", "策略熵 (Entropy)", "entropy"),
        ("response_length/mean", "响应长度", "length"),
    ]

    # 配色方案 - 更鲜明的颜色
    colors = {
        "k3v4_nc": "#E74C3C",  # 红色
        "k2v3_nc": "#3498DB",  # 蓝色
        "int4_per_token_head": "#2ECC71",  # 绿色
    }

    # 中文标签
    scheme_labels = {
        "k3v4_nc": "K3V4 (3bit key, 4bit value)",
        "k2v3_nc": "K2V3 (2bit key, 3bit value)",
        "int4_per_token_head": "INT4 per-token-head",
    }

    # 创建 3x2 子图
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("1.5B QAT 训练曲线对比", fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for idx, (metric_key, title, metric_type) in enumerate(key_metrics):
        ax = axes[idx]

        # 绘制三臂数据
        for scheme in ["k3v4_nc", "k2v3_nc", "int4_per_token_head"]:
            if scheme not in metrics_data:
                continue

            if metric_key not in metrics_data[scheme]:
                continue

            data = metrics_data[scheme][metric_key]
            steps = [d['step'] for d in data]
            values = [d['value'] for d in data]

            # 绘制曲线
            ax.plot(steps, values,
                   label=scheme_labels[scheme],
                   color=colors[scheme],
                   linewidth=2.5,
                   alpha=0.9)

        # 设置标题和标签
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("训练步数 (Step)", fontsize=10)
        ax.set_ylabel("值", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)

        # Y 轴范围优化
        if metric_type == "overlap":
            ax.set_ylim([0.85, 0.95])
        elif metric_type == "align":
            ax.set_ylim([0, 0.15])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"训练曲线已保存到: {output_path}")

def plot_block_align_comparison(metrics_data, output_path):
    """绘制 block_align 的详细对比（替代逐层 kv_align）"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Block Align (NMSE) 详细分析", fontsize=14, fontweight='bold')

    colors = {
        "k3v4_nc": "#E74C3C",
        "k2v3_nc": "#3498DB",
        "int4_per_token_head": "#2ECC71",
    }

    scheme_labels = {
        "k3v4_nc": "K3V4",
        "k2v3_nc": "K2V3",
        "int4_per_token_head": "INT4",
    }

    # 1. Block align 主曲线
    ax = axes[0, 0]
    for scheme in ["k3v4_nc", "k2v3_nc", "int4_per_token_head"]:
        if "block_align" in metrics_data.get(scheme, {}):
            data = metrics_data[scheme]["block_align"]
            steps = [d['step'] for d in data]
            values = [d['value'] for d in data]
            ax.plot(steps, values, label=scheme_labels[scheme],
                   color=colors[scheme], linewidth=2.5)

    ax.set_title("Block Align (总体)", fontweight='bold')
    ax.set_xlabel("Step")
    ax.set_ylabel("NMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Validation top-k overlap
    ax = axes[0, 1]
    for scheme in ["k3v4_nc", "k2v3_nc", "int4_per_token_head"]:
        if "val-topk/overlap_ratio" in metrics_data.get(scheme, {}):
            data = metrics_data[scheme]["val-topk/overlap_ratio"]
            steps = [d['step'] for d in data]
            values = [d['value'] for d in data]
            ax.plot(steps, values, label=scheme_labels[scheme],
                   color=colors[scheme], linewidth=2.5)

    ax.set_title("Top-K 重叠率 (验证集)", fontweight='bold')
    ax.set_xlabel("Step")
    ax.set_ylabel("Overlap Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 0.95])

    # 3. Policy loss
    ax = axes[1, 0]
    for scheme in ["k3v4_nc", "k2v3_nc", "int4_per_token_head"]:
        if "actor/policy_loss" in metrics_data.get(scheme, {}):
            data = metrics_data[scheme]["actor/policy_loss"]
            steps = [d['step'] for d in data]
            values = [d['value'] for d in data]
            ax.plot(steps, values, label=scheme_labels[scheme],
                   color=colors[scheme], linewidth=2.5)

    ax.set_title("Policy Loss", fontweight='bold')
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Reward
    ax = axes[1, 1]
    for scheme in ["k3v4_nc", "k2v3_nc", "int4_per_token_head"]:
        if "actor/reward" in metrics_data.get(scheme, {}):
            data = metrics_data[scheme]["actor/reward"]
            steps = [d['step'] for d in data]
            values = [d['value'] for d in data]
            ax.plot(steps, values, label=scheme_labels[scheme],
                   color=colors[scheme], linewidth=2.5)

    ax.set_title("奖励 (Reward)", fontweight='bold')
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Block align 分析图已保存到: {output_path}")

def main():
    # 三臂运行信息
    runs_info = [
        ("k3v4_nc", 12392,
         "/historical/shared/qat/runs/qat-train/qwen_brorl_1p5b/turboquant_k3v4_nc-ba1.0-g2-b64-lr1.5e-6-k512-union-1p5b-nmseopd-pl512-j12392",
         "/historical/shared/qat/r/qat-train/train-12392.out"),
        ("k2v3_nc", 12393,
         "/historical/shared/qat/runs/qat-train/qwen_brorl_1p5b/turboquant_k2v3_nc-ba1.0-g2-b64-lr1.5e-6-k512-union-1p5b-nmseopd-pl512-j12393",
         "/historical/shared/qat/r/qat-train/train-12393.out"),
        ("int4_per_token_head", 12954,
         "/historical/shared/qat/runs/qat-train/qwen_brorl_1p5b/int4_per_token_head-ba1.0-g2-b64-lr1.5e-6-k512-union-1p5b-nmseopd-pl512-j12954",
         "/historical/shared/qat/r/qat-train/train-12954.out"),
    ]

    # 输出目录
    output_dir = Path("/historical/home/QAT/experiments/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("生成改进版训练报告")
    print("=" * 60)

    # 1. 生成 settings 文档
    print("\n1. 生成配置文档...")
    settings_md = create_settings_markdown(runs_info)
    settings_path = output_dir / "current_training_settings.md"
    with open(settings_path, 'w') as f:
        f.write(settings_md)
    print(f"   配置文档已保存: {settings_path}")

    # 2. 加载训练指标
    print("\n2. 加载训练指标...")
    metrics_path = "/historical/home/QAT/tmp_training_metrics.json"
    metrics_data = load_metrics(metrics_path)

    for scheme in metrics_data:
        n_metrics = len(metrics_data[scheme])
        n_points = sum(len(v) for v in metrics_data[scheme].values())
        print(f"   {scheme}: {n_metrics} 个指标, {n_points} 个数据点")

    # 3. 生成训练曲线图
    print("\n3. 生成高质量训练曲线...")
    training_curves_path = output_dir / "training_curves_improved.png"
    plot_training_curves(metrics_data, training_curves_path)

    # 4. 生成 block_align 详细分析图
    print("\n4. 生成 block_align 详细分析...")
    block_align_path = output_dir / "block_align_analysis.png"
    plot_block_align_comparison(metrics_data, block_align_path)

    print("\n" + "=" * 60)
    print("报告生成完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - 配置文档: {settings_path}")
    print(f"  - 训练曲线: {training_curves_path}")
    print(f"  - Block align 分析: {block_align_path}")
    print(f"\n说明:")
    print(f"  - 问题 #1 (settings 不清晰): 已解决，见 current_training_settings.md")
    print(f"  - 问题 #2 (绘图质量差): 已解决，使用更清晰的配色、字体和布局")
    print(f"  - 问题 #3 (逐层 kv_align): 当前训练 kv_align_coef=0，无逐层数据")
    print(f"    替代方案: block_align_analysis.png 展示了 block_align (NMSE) 的详细对比")

if __name__ == "__main__":
    main()
