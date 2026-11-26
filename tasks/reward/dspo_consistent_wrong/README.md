# AutoRLBench Task: dspo_consistent_wrong

- **Category**: `reward`
- **Level**: issue
- **One-liner**: consensus reward 不涨(became_correct≈0)，根因是整组都错、正确答案不在组里，negative filtering 翻不了盘

## 诊断陷阱
以为 reward 需要调参，实际是结构问题：组内 coverage≈0

## 根因（oracle 摘要）
整组 G 条 rollout 都错，正确答案不在组里；negative filtering 只能重排错答案，不能 flip wrong→correct；天花板=组内 coverage

## 构造
`evaluations/solution/oracle/simulate_training.py` 生成 frozen_evidence（模拟 run，自洽可复现）。
