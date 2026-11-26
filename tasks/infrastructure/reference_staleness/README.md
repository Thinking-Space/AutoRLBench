# reference_staleness

- **Category**: `infrastructure`
- **One-liner**: KL 惩罚失效，根因是 reference model 没更新(stale)

## 陷阱
以为 KL 系数问题，实际是 reference 没更新

## 根因
reference model 停留在初始 checkpoint→KL 惩罚基准错误→policy 漂移

## 构造
`evaluations/solution/oracle/simulate_training.py`
