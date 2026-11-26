# kl_explosion

- **Category**: `algorithm`
- **One-liner**: KL 散度爆炸训练发散，根因是 KL 系数太小 policy 漂移过远

## 陷阱
以为学习率太大，实际是 KL 约束失效

## 根因
KL 系数 0.001 太小，policy 漂移过远，KL 爆炸→reward 崩

## 构造
`evaluations/solution/oracle/simulate_training.py`
