# Issue 诊断

## 背景
KL 惩罚不生效、policy 漂移。同目录提供 metrics。

## 任务
1. 读 `../training_report.txt` 与 `../metrics_summary.json`。
2. 用 `tools/inspect_logs.py` 检查。
3. 定位根因+证据。
4. 输出诊断。
