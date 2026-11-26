# AutoRLBench Task: tail_self_reinforcement

- **Category**: `algorithm`
- **Level**: issue
- **One-liner**: 弱任务层越训越弱，根因是 age 过滤+优先调度→富者愈富，gap 越拉越大

## 诊断陷阱
以为是探索不足，实际是 age 过滤的富者愈富

## 根因（oracle 摘要）
age 过滤让弱任务(长/慢)拿不到梯度→更弱→更少被采→富者愈富

## 构造
`evaluations/solution/oracle/simulate_training.py` 生成 frozen_evidence（模拟 run，自洽可复现）。
