# AutoRLBench Task: lag_critic_variance_explosion

- **Category**: `algorithm`
- **Level**: issue
- **One-liner**: 多步 advantage 估计方差爆炸，根因是重要性权重连乘随 horizon 指数增长

## 诊断陷阱
以为学习率太大，实际是重要性权重连乘的指数方差

## 根因（oracle 摘要）
多步估计量重要性权重连乘，方差随 horizon 指数增长；capping 引入偏差

## 构造
`evaluations/solution/oracle/simulate_training.py` 生成 frozen_evidence（模拟 run，自洽可复现）。
