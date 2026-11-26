# Issue 诊断指令（给被测 agent）

## 背景
弱任务层成功率随训练下降，gap 扩大。同目录提供 metrics 与配置。

## 你的任务
1. 阅读 `../training_report.txt` 与 `../metrics_summary.json`。
2. 用 `tools/inspect_logs.py` 检查数据。
3. **定位根因**并给出证据。
4. 输出诊断报告。

## 规则
- 只依据本 workspace 材料。
- 注意区分"方法不行"与"配置/结构问题"。
