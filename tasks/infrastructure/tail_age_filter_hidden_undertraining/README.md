# AutoRLBench Task: tail_age_filter_hidden_undertraining

- **Category**: `infrastructure`
- **Level**: issue
- **Domain**: 异步 RL（GRPO 类）+ age 准入门控，长程 agent 任务
- **One-liner**: age 门控按 wall-time 过滤长任务 → 长程能力隐形欠训练；聚合成功率(短任务主导)看不出问题，只有按 horizon 切开才暴露长程塌陷。

## 诊断陷阱
聚合成功率紧门控 0.954 vs 放宽 0.999（几乎一样），短任务成功率 1.000（正常）——
常规指标完全看不出问题。必须按 horizon 分层看，才发现长程段(h>10)从 0.869 塌到 0.003。

## 根因（oracle 摘要）
异步 age 准入门控：rollout 落后超过 age_limit 个版本则丢弃。长任务 wall-time 长 → lag 大 →
被系统性过滤 → 模型很少练到长程行为。聚合指标被短任务主导，掩盖了长程塌陷。

## 构造
`evaluations/solution/oracle/simulate_training.py` 生成 frozen_evidence（模拟 run，自洽可复现）。