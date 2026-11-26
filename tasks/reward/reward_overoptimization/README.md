# reward_overoptimization

- **Category**: `reward`
- **One-liner**: proxy reward 涨但 true reward 跌，根因是 reward model 过优化

## 陷阱
以为 proxy reward 涨就是好了

## 根因
reward model 过优化→proxy reward 涨但 true reward 跌(Goodhart)

## 构造
`evaluations/solution/oracle/simulate_training.py`
