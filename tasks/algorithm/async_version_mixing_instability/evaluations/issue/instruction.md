# Issue 诊断指令（给被测 agent）

## 背景
一个用 **GRPO + 异步 rollout** 训练数学推理模型的 run（verifier 给 0/1 奖励）比同配置的**同步参考 run** 收敛到明显更差的水平。同目录提供同步与异步两个 run 的冻结 rollout 日志。

## 你的任务
1. 阅读 `../training_report.txt` 与 `../metrics_summary.json`（如有）。
2. 用 `tools/inspect_logs.py` 检查原始日志（`frozen_evidence/`）。
3. **定位异步 run 学得更差的根因**，并给出支撑证据。
4. 输出一份诊断报告，说明：根因、你用什么指标/证据确认、以及（可选）修复方向。

## 规则
- 只依据本 workspace 提供的材料，不要假设额外信息。
- 结论要可验证：给出你实际计算的指标或检查过的数据结构。
