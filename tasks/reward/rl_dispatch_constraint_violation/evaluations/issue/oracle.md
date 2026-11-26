# Oracle：根因与证据

## 根因
vanilla/cvar 的 reward 只含 cost 不含约束，viol_rate=1.0；safe(Lagrangian) viol_rate=0.0

## 诊断陷阱
以为调 cost 权重就行，实际约束没进 reward

## 证据
viol_rate: vanilla 1.0 / cvar 1.0 / safe 0.0

## 判分要点
- 根因必须落到上述根因，不能只归因表面症状。
- 证据必须来自上述证据。
- 只结论表面症状 → Fail。
