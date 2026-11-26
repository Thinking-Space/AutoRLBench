# reward_hacking_length

- **Category**: `reward`
- **One-liner**: 模型靠生成更长答案 hack reward，根因是 length bonus 奖励了长度而非正确性

## 陷阱
以为模型变强了，实际是 reward hacking

## 根因
length bonus 让模型学会写长答案骗 reward；真实准确率不变

## 构造
`evaluations/solution/oracle/simulate_training.py`
