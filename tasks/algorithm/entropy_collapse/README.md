# entropy_collapse

- **Category**: `algorithm`
- **One-liner**: 熵坍缩失去探索，根因是 entropy bonus 太小+温度太低

## 陷阱
以为模型收敛了，实际是熵坍缩

## 根因
entropy bonus 0.001 太小+温度 0.7 太低→熵坍缩→失去探索→reward 停滞

## 构造
`evaluations/solution/oracle/simulate_training.py`
