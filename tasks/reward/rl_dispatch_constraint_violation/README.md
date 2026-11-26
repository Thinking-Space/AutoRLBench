# AutoRLBench Task: rl_dispatch_constraint_violation

- **Category**: `reward`
- **Level**: issue
- **One-liner**: RL 调度违约率 100%，根因是约束没进 reward 只有 cost，需 Lagrangian safe RL

## 诊断陷阱
以为调 cost 权重就行，实际约束没进 reward

## 根因（oracle 摘要）
vanilla/cvar 的 reward 只含 cost 不含约束，viol_rate=1.0；safe(Lagrangian) viol_rate=0.0

## 构造
`evaluations/solution/oracle/simulate_training.py` 生成 frozen_evidence（模拟 run，自洽可复现）。
