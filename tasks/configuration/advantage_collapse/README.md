# advantage_collapse

- **Category**: `configuration`
- **One-liner**: GRPO 组内全对或全错→advantage=0→无梯度，根因是 group size 太小+难度不匹配

## 陷阱
以为学习率太小，实际是组内无信号

## 根因
group size 4 太小+题目太难→组内全错→advantage 全 0→无梯度

## 构造
`evaluations/solution/oracle/simulate_training.py`
