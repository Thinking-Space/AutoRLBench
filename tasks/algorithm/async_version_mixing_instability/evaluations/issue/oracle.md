# Oracle：根因与证据（issue 级）

## 根因
异步 rollout 导致同一个 GRPO 组内**混入多个 policy 版本**：
- `behavior_version` 在不同 rollout 间不同（组内 spread≈2.2，**98% 的组混版本**；同步=0）。
- GRPO 的组内标准化 `A_i=(r_i-mean)/std` 隐含要求组内同策略；版本混合后，
  `mean` 不再是单一策略的条件基线，advantage 同时混入"动作质量"与"版本强弱差异"，
  credit assignment 被污染 → 学习效率变差。

## 证据（给分依据）
| 项 | 期望值 |
|---|---|
| 组内 behavior_version spread | sync=0.00 / async≈2.2 |
| 含混合版本的组占比 | async≈98% |
| 辅助信号 offset_fraction | sync≈0.00 / async≈0.6（0/1 奖励下有估计伪影，仅方向参考） |
| 陷阱：normalized advantage variance | sync≈0.95 / async≈0.90（无法区分，**不作为证据**） |
| 陷阱：reward vs lag 相关性 | ≈0（会误导，不作为证据） |

## 判分要点
- 根因必须落到"**组内混版本 → group baseline 比较不同强度策略 → 污染 advantage**"。
- 证据必须来自**组构成（behavior_version spread / 版本混合占比）**。
- 只停留在 normalized advantage variance / reward 曲线等常规标量 → 证据不成立。
