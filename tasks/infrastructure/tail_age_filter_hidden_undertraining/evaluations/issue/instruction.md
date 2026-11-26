# Issue 诊断指令（给被测 agent）

## 背景
一个异步 RL 训练（GRPO 类，带 age 准入门控：rollout 落后超过 age_limit 个版本则丢弃）用于长程 agent 任务。训练看起来正常，聚合成功率与放宽门控的对照几乎一样。同目录提供紧门控与放宽门控的按-horizon 成功率、以及真实 horizon 分布。

## 你的任务
1. 阅读 `../training_report.txt` 与 `../metrics_summary.json`。
2. 用 `tools/inspect_logs.py` 检查按 horizon 的成功率，阅读 `frozen_evidence/horizon_dist.json`。
3. **判断这个异步 RL 训练是否被 age 门控隐形损害了某些能力**；如果有，是什么、用什么证据发现。
4. 输出诊断报告。

## 规则
- 只依据本 workspace 提供的材料。
- 结论要可验证：给出你实际分层/对比的指标。