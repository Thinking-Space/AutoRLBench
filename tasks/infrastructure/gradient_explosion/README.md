# gradient_explosion

- **Category**: `infrastructure`
- **One-liner**: 梯度范数尖峰训练发散，根因是 grad clip 阈值太大+batch 太小

## 陷阱
以为模型不兼容，实际是梯度爆炸

## 根因
grad clip 10.0 太大+batch 8 太小→梯度尖峰→loss NaN

## 构造
`evaluations/solution/oracle/simulate_training.py`
