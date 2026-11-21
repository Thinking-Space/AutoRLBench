#!/usr/bin/env python3
"""生成改进版的训练报告（HTML + 纯文本，不依赖 matplotlib）"""

import json
from pathlib import Path
from collections import defaultdict

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

def get_metric_stats(data):
    """计算指标统计信息"""
    if not data:
        return {}

    values = [d['value'] for d in data]
    steps = [d['step'] for d in data]

    return {
        'min': min(values),
        'max': max(values),
        'latest': values[-1],
        'first': values[0],
        'n_points': len(values),
        'latest_step': steps[-1],
    }

def create_html_report(metrics_data, runs_info, output_path):
    """创建 HTML 交互式报告"""

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>1.5B QAT 训练报告</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }
        h3 { color: #7f8c8d; }
        .settings-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }
        .settings-card { background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #3498db; }
        .settings-card h3 { margin-top: 0; color: #2c3e50; font-size: 14px; }
        .settings-table { width: 100%; font-size: 12px; }
        .settings-table td { padding: 4px 8px; }
        .settings-table td:first-child { font-weight: 600; color: #555; width: 60%; }
        .settings-table td:last-child { color: #2c3e50; font-family: 'Courier New', monospace; background: #fff; padding: 4px 8px; border-radius: 3px; }
        .chart { margin: 20px 0; }
        .stats-box { background: #ecf0f1; padding: 15px; border-radius: 6px; margin: 10px 0; }
        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .stat-item { text-align: center; padding: 10px; background: white; border-radius: 4px; }
        .stat-label { font-size: 11px; color: #7f8c8d; text-transform: uppercase; }
        .stat-value { font-size: 18px; font-weight: bold; color: #2c3e50; }
        .scheme-k3v4 { color: #e74c3c; }
        .scheme-k2v3 { color: #3498db; }
        .scheme-int4 { color: #2ecc71; }
        .note { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; border-radius: 4px; }
        .note strong { color: #856404; }
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 1.5B QAT 训练报告（改进版）</h1>
    <p><strong>生成时间:</strong> 2026-09-08 | <strong>训练任务:</strong> 三臂 KV 量化对比</p>
"""

    # 添加配置对比
    html += "<h2>📋 训练配置对比</h2>\n"
    html += '<div class="settings-grid">\n'

    for scheme, jobid, run_dir, _ in runs_info:
        settings = load_settings(run_dir)
        summary = format_settings_summary(settings)

        scheme_class = scheme.replace('_', '-')
        html += f'<div class="settings-card">\n'
        html += f'<h3 class="scheme-{scheme_class}">● {scheme} (Job {jobid})</h3>\n'
        html += '<table class="settings-table">\n'

        for section, params in summary.items():
            for key, value in params.items():
                html += f'<tr><td>{key}</td><td>{value}</td></tr>\n'

        html += '</table>\n</div>\n'

    html += '</div>\n'

    # 关键指标定义
    key_metrics = [
        ("actor/policy_loss", "Policy Loss"),
        ("actor/reward", "奖励 (Reward)"),
        ("val-topk/overlap_ratio", "Top-K 重叠率"),
        ("block_align", "Block Align (NMSE)"),
        ("actor/entropy", "策略熵"),
        ("response_length/mean", "响应长度"),
    ]

    # 为每个指标创建图表
    html += "<h2>📊 训练曲线</h2>\n"

    for metric_key, title in key_metrics:
        html += f'<div class="chart">\n'
        html += f'<h3>{title}</h3>\n'
        html += f'<div id="chart-{metric_key.replace("/", "-")}" style="width:100%;height:400px;"></div>\n'

        # 生成 Plotly 数据
        traces = []
        for scheme in ["k3v4_nc", "k2v3_nc", "int4_per_token_head"]:
            if scheme not in metrics_data or metric_key not in metrics_data[scheme]:
                continue

            data = metrics_data[scheme][metric_key]
            steps = [d['step'] for d in data]
            values = [d['value'] for d in data]

            color_map = {
                "k3v4_nc": "#e74c3c",
                "k2v3_nc": "#3498db",
                "int4_per_token_head": "#2ecc71",
            }

            label_map = {
                "k3v4_nc": "K3V4 (3bit key, 4bit value)",
                "k2v3_nc": "K2V3 (2bit key, 3bit value)",
                "int4_per_token_head": "INT4 per-token-head",
            }

            trace = f"""{{
                x: {json.dumps(steps)},
                y: {json.dumps(values)},
                name: '{label_map[scheme]}',
                type: 'scatter',
                mode: 'lines',
                line: {{color: '{color_map[scheme]}', width: 2.5}}
            }}"""
            traces.append(trace)

        if traces:
            html += f"""
<script>
    var data = [{','.join(traces)}];
    var layout = {{
        title: '',
        xaxis: {{title: '训练步数 (Step)'}},
        yaxis: {{title: '值'}},
        hovermode: 'x unified',
        plot_bgcolor: '#f8f9fa',
        paper_bgcolor: 'white',
    }};
    Plotly.newPlot('chart-{metric_key.replace("/", "-")}', data, layout, {{responsive: true}});
</script>
"""

        html += '</div>\n'

    # 添加统计摘要
    html += "<h2>📈 当前训练状态摘要</h2>\n"

    for scheme in ["k3v4_nc", "k2v3_nc", "int4_per_token_head"]:
        if scheme not in metrics_data:
            continue

        html += f'<div class="stats-box">\n'
        html += f'<h3 class="scheme-{scheme.replace("_", "-")}">● {scheme}</h3>\n'
        html += '<div class="stats-grid">\n'

        # 选择几个关键指标展示
        for metric_key, title in [("actor/policy_loss", "Policy Loss"),
                                   ("val-topk/overlap_ratio", "Top-K Overlap"),
                                   ("block_align", "Block Align")]:
            if metric_key in metrics_data[scheme]:
                stats = get_metric_stats(metrics_data[scheme][metric_key])
                html += f"""
<div class="stat-item">
    <div class="stat-label">{title}</div>
    <div class="stat-value">{stats['latest']:.4f}</div>
    <div class="stat-label">Step {stats['latest_step']}</div>
</div>
"""

        html += '</div>\n</div>\n'

    # 添加说明
    html += """
<div class="note">
    <strong>⚠️ 关于逐层 kv_align 分析:</strong><br>
    当前三臂训练的 <code>kv_align_coef=0.0</code>，kv_align 损失项被禁用。<br>
    虽然代码中有逐层计算能力（<code>qat_kv/kv_align.py</code>），但因系数为0，没有逐层数据被记录。<br>
    当前使用 <code>block_align_coef=1.0</code> (NMSE) + OPD 奖励的组合训练。<br>
    如需逐层 kv_align 分析，需启动新的训练臂设置 <code>kv_align_coef > 0</code>。
</div>

<h2>✅ 改进说明</h2>
<ul>
    <li><strong>问题 #1 (settings 不清晰):</strong> 已解决 - 顶部配置对比卡片清晰展示三臂核心配置</li>
    <li><strong>问题 #2 (绘图质量差):</strong> 已解决 - 使用 Plotly 交互式图表，支持缩放、悬停查看数值</li>
    <li><strong>问题 #3 (缺少逐层 kv_align):</strong> 当前训练无此数据，见上方说明</li>
</ul>
"""

    html += """
</div>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"HTML 报告已保存: {output_path}")

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
    print("生成改进版训练报告（HTML）")
    print("=" * 60)

    # 1. 生成 settings 文档
    print("\n1. 生成配置文档...")
    settings_md = create_settings_markdown(runs_info)
    settings_path = output_dir / "current_training_settings.md"
    with open(settings_path, 'w', encoding='utf-8') as f:
        f.write(settings_md)
    print(f"   ✓ 配置文档: {settings_path}")

    # 2. 加载训练指标
    print("\n2. 加载训练指标...")
    metrics_path = "/historical/home/QAT/tmp_training_metrics.json"
    metrics_data = load_metrics(metrics_path)

    for scheme in metrics_data:
        n_metrics = len(metrics_data[scheme])
        n_points = sum(len(v) for v in metrics_data[scheme].values())
        print(f"   {scheme}: {n_metrics} 个指标, {n_points} 个数据点")

    # 3. 生成 HTML 报告
    print("\n3. 生成交互式 HTML 报告...")
    html_path = output_dir / "training_report_improved.html"
    create_html_report(metrics_data, runs_info, html_path)

    print("\n" + "=" * 60)
    print("报告生成完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - 配置文档: {settings_path}")
    print(f"  - HTML 报告: {html_path}")
    print(f"\n说明:")
    print(f"  ✓ 问题 #1 (settings 不清晰): 已解决")
    print(f"  ✓ 问题 #2 (绘图质量差): 已解决，使用 Plotly 交互式图表")
    print(f"  ⚠ 问题 #3 (逐层 kv_align): 当前训练 kv_align_coef=0，无数据")

if __name__ == "__main__":
    main()
