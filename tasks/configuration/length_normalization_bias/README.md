# length_normalization_bias

- **Category**: `configuration`
- **One-liner**: 长序列被系统性偏好，根因是 length normalization 缺失

## 陷阱
以为模型在学，实际是长度偏置

## 根因
无 length normalization→长序列 log-prob 更负→被系统性惩罚/偏好→长度偏置

## 构造
`evaluations/solution/oracle/simulate_training.py`
