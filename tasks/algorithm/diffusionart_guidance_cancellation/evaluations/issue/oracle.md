# Oracle：根因与证据

## 根因
guidance 梯度 ρ_guid≈0.024 几乎全抵消；restoration 存活 ρ_rest≈0.474；需 NTK 预条件 restoration

## 诊断陷阱
以为学习率/步数不够，实际是 guidance 梯度被逐状态 Jacobian 门控抵消

## 证据
按通路测梯度相干性 ρ：guidance 0.024 vs restoration 0.474

## 判分要点
- 根因必须落到上述根因，不能只归因表面症状。
- 证据必须来自上述证据。
- 只结论表面症状 → Fail。
