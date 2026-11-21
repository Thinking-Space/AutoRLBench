#!/usr/bin/env python3
"""生成全面的训练报告：overall历史 + 当前训练 + 逐层分析"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def collect_all_experiments():
    """收集所有历史实验配置"""
    runs_dir = Path("/historical/shared/qat/runs/qat-train")
    experiments = []

    for settings_file in runs_dir.rglob("effective_settings.json"):
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)

            run_dir = settings_file.parent

            # 提取关键信息
            exp_info = {
                'run_dir': str(run_dir),
                'profile': settings.get('profile', 'unknown'),
                'kv_scheme': settings.get('kv_scheme', 'N/A'),
                'block_align_coef': settings.get('block_align_coef', 0),
                'kv_align_coef': settings.get('kv_align_coef', 0),
                'objective': settings.get('objective', 'N/A'),
                'updates': settings.get('updates', 0),
                'actor_lr': settings.get('actor_lr', 0),
                'global_batch': settings.get('global_batch', 0),
                'reward_top_k': settings.get('reward_top_k', 0),
                'teacher_routing': settings.get('teacher_routing', 'N/A'),
                'rollout_kv_quantized': settings.get('rollout_kv_quantized', False),
                'job_id': _extract_job_id(str(run_dir)),
                'timestamp': settings_file.stat().st_mtime,
            }

            experiments.append(exp_info)

        except Exception as e:
            print(f"警告: 跳过 {settings_file}: {e}", file=sys.stderr)
            continue

    # 按时间排序
    experiments.sort(key=lambda x: x['timestamp'], reverse=True)

    return experiments

def _extract_job_id(run_dir_str):
    """从路径提取job ID"""
    import re
    match = re.search(r'-j(\d+)', run_dir_str)
    return match.group(1) if match else 'unknown'

def group_experiments_by_profile(experiments):
    """按profile分组实验"""
    grouped = defaultdict(list)
    for exp in experiments:
        grouped[exp['profile']].append(exp)
    return grouped

def generate_html_report(experiments, output_path):
    """生成完整的HTML报告"""

    # 按profile分组
    grouped = group_experiments_by_profile(experiments)

    # 当前运行（最新的3个1.5B实验）
    current_runs = [exp for exp in experiments if 'qwen_brorl_1p5b' in exp['run_dir']][:3]

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QAT 训练全面报告</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #2c3e50;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 36px; margin-bottom: 10px; font-weight: 700; }
        .header p { font-size: 16px; opacity: 0.9; }

        .content { padding: 40px; }

        .section { margin-bottom: 50px; }
        .section-title {
            font-size: 28px;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        /* Overall 历史统计 */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .stat-value { font-size: 42px; font-weight: 700; margin-bottom: 5px; }
        .stat-label { font-size: 14px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; }

        /* 实验表格 */
        .exp-table-container {
            overflow-x: auto;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
        }
        thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        th {
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
            font-size: 13px;
        }
        tbody tr:hover {
            background: #f8f9fa;
        }

        /* Profile 分组 */
        .profile-section {
            margin: 30px 0;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
        }
        .profile-header {
            font-size: 20px;
            font-weight: 600;
            color: #667eea;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .profile-badge {
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }

        /* 当前训练对比 */
        .current-runs {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .run-card {
            background: white;
            border: 2px solid #e1e8ed;
            border-radius: 10px;
            padding: 20px;
            transition: all 0.3s ease;
        }
        .run-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            border-color: #667eea;
        }
        .run-card-header {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }
        .run-card-body {
            font-size: 13px;
            line-height: 1.8;
        }
        .run-card-body .param {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .run-card-body .param-key {
            color: #7f8c8d;
            font-weight: 500;
        }
        .run-card-body .param-value {
            font-weight: 600;
            color: #2c3e50;
            font-family: 'Courier New', monospace;
        }

        /* 配色 */
        .scheme-k3v4 { color: #e74c3c; }
        .scheme-k2v3 { color: #3498db; }
        .scheme-int4 { color: #2ecc71; }

        /* 图表区域 */
        .chart-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        /* 注释框 */
        .note {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .note strong { color: #856404; }

        /* 响应式 */
        @media (max-width: 768px) {
            .container { margin: 10px; }
            .content { padding: 20px; }
            .stats-grid { grid-template-columns: 1fr; }
            .current-runs { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🔬 QAT 训练全面报告</h1>
        <p>Overall 历史实验概览 + 当前训练详情 + 逐层分析</p>
        <p style="font-size: 14px; margin-top: 10px; opacity: 0.8;">生成时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>

    <div class="content">
"""

    # ==================== Overall 统计 ====================
    html += """
        <div class="section">
            <div class="section-title">📊 Overall 训练历史统计</div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">""" + str(len(experiments)) + """</div>
                    <div class="stat-label">总实验数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">""" + str(len(grouped)) + """</div>
                    <div class="stat-label">不同 Profile</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">""" + str(len([e for e in experiments if e['rollout_kv_quantized']])) + """</div>
                    <div class="stat-label">量化 Rollout</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">""" + str(len(set(e['kv_scheme'] for e in experiments))) + """</div>
                    <div class="stat-label">不同 KV 方案</div>
                </div>
            </div>
"""

    # ==================== 按 Profile 分组展示 ====================
    html += """
            <div class="profile-section">
                <div class="profile-header">
                    <span>🗂️ 按 Profile 分组</span>
                </div>
"""

    for profile, exps in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
        html += f"""
                <details style="margin: 15px 0;">
                    <summary style="cursor: pointer; font-weight: 600; padding: 10px; background: white; border-radius: 6px; margin-bottom: 10px;">
                        <span style="color: #667eea;">▶</span> {profile}
                        <span class="profile-badge">{len(exps)} 个实验</span>
                    </summary>
                    <div class="exp-table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Job ID</th>
                                    <th>KV Scheme</th>
                                    <th>Block Align</th>
                                    <th>KV Align</th>
                                    <th>Objective</th>
                                    <th>Updates</th>
                                    <th>LR</th>
                                    <th>Batch</th>
                                    <th>Quantized Rollout</th>
                                </tr>
                            </thead>
                            <tbody>
"""
        for exp in exps[:20]:  # 每个profile最多显示20个
            html += f"""
                                <tr>
                                    <td><code>{exp['job_id']}</code></td>
                                    <td><strong>{exp['kv_scheme']}</strong></td>
                                    <td>{exp['block_align_coef']}</td>
                                    <td>{exp['kv_align_coef']}</td>
                                    <td><small>{exp['objective']}</small></td>
                                    <td>{exp['updates']}</td>
                                    <td>{exp['actor_lr']}</td>
                                    <td>{exp['global_batch']}</td>
                                    <td>{'✓' if exp['rollout_kv_quantized'] else '✗'}</td>
                                </tr>
"""

        html += """
                            </tbody>
                        </table>
                    </div>
                </details>
"""

    html += """
            </div>
        </div>
"""

    # ==================== 当前训练详情 ====================
    html += """
        <div class="section">
            <div class="section-title">🚀 当前训练详情 (1.5B 三臂)</div>

            <div class="current-runs">
"""

    color_map = {
        'k3v4_nc': '#e74c3c',
        'k2v3_nc': '#3498db',
        'int4_per_token_head': '#2ecc71',
    }

    for exp in current_runs:
        scheme = exp['kv_scheme']
        color = color_map.get(scheme, '#95a5a6')

        html += f"""
                <div class="run-card">
                    <div class="run-card-header" style="color: {color};">
                        ● {scheme}
                        <span style="float: right; font-size: 12px; color: #7f8c8d;">Job {exp['job_id']}</span>
                    </div>
                    <div class="run-card-body">
                        <div class="param">
                            <span class="param-key">KV 方案</span>
                            <span class="param-value">{scheme}</span>
                        </div>
                        <div class="param">
                            <span class="param-key">Block Align 系数</span>
                            <span class="param-value">{exp['block_align_coef']}</span>
                        </div>
                        <div class="param">
                            <span class="param-key">KV Align 系数</span>
                            <span class="param-value">{exp['kv_align_coef']}</span>
                        </div>
                        <div class="param">
                            <span class="param-key">目标函数</span>
                            <span class="param-value">{exp['objective']}</span>
                        </div>
                        <div class="param">
                            <span class="param-key">训练步数</span>
                            <span class="param-value">{exp['updates']}</span>
                        </div>
                        <div class="param">
                            <span class="param-key">学习率</span>
                            <span class="param-value">{exp['actor_lr']}</span>
                        </div>
                        <div class="param">
                            <span class="param-key">批大小</span>
                            <span class="param-value">{exp['global_batch']}</span>
                        </div>
                        <div class="param">
                            <span class="param-key">OPD Top-K</span>
                            <span class="param-value">{exp['reward_top_k']}</span>
                        </div>
                        <div class="param">
                            <span class="param-key">量化 Rollout</span>
                            <span class="param-value">{'✓ 是' if exp['rollout_kv_quantized'] else '✗ 否'}</span>
                        </div>
                    </div>
                </div>
"""

    html += """
            </div>
        </div>
"""

    # ==================== 注释说明 ====================
    html += """
        <div class="section">
            <div class="note">
                <strong>📝 关于逐层 KV Align 分析:</strong><br>
                当前三臂训练的 <code>kv_align_coef=0.0</code>，kv_align 损失项被禁用。<br>
                需要从 checkpoint (step 30, 60, 90...) 进行前向推理来提取逐层数据。<br>
                正在准备脚本进行提取...
            </div>
        </div>
"""

    html += """
    </div>
</div>
</body>
</html>
"""

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ HTML 报告已生成: {output_path}")

def main():
    print("=" * 60)
    print("生成全面训练报告")
    print("=" * 60)

    print("\n1. 收集所有历史实验...")
    experiments = collect_all_experiments()
    print(f"   找到 {len(experiments)} 个实验")

    print("\n2. 生成 HTML 报告...")
    output_dir = Path("/historical/home/QAT/experiments/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "comprehensive_report.html"
    generate_html_report(experiments, html_path)

    print("\n" + "=" * 60)
    print("报告生成完成!")
    print("=" * 60)
    print(f"\n输出文件: {html_path}")
    print(f"\n包含:")
    print(f"  ✓ Overall 历史统计 ({len(experiments)} 个实验)")
    print(f"  ✓ 按 Profile 分组展示")
    print(f"  ✓ 当前训练详情对比")
    print(f"  ⏳ 逐层分析 (需要从 checkpoint 提取)")

if __name__ == "__main__":
    main()
