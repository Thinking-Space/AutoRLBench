# Oracle：根因与证据（issue 级）

## 根因（两个叠加）
1. **EMA/student 评错权重**：训练用 EMA 采样，却评 student 权重 → RL 臂看起来全低于 SFT；
   评 EMA 权重则 ≥SFT（B1 0.46 / A-gae 0.48）。
2. **训练崩溃**：nft_weight_scale 把 mse 形式 [1.0, 0.2] 抄进 dpo 形式 → 能量对称违反 →
   B1 几步内成功率归零、自锁（step5=0.04, step100=0.0）。

## 诊断陷阱
三组 RL 全低于 SFT、A-gae 未超 A-uniform → 容易误判"credit assignment 方法不行"。
实际是评错权重 + 训练崩溃，credit 问题根本没被干净测到。

## 证据（给分依据）
| 项 | 期望值 |
|---|---|
| student 评测 | B1 0.22 / A-gae 0.344 / A-uniform 0.374（全<SFT 0.44） |
| EMA 评测 | B1 0.46 / A-gae 0.48（≥SFT） |
| B1 训练曲线 | step5≈0.04, step100=0.0（崩溃自锁） |
| 配置 | nft_weight_scale [1.0,0.2] mse 形式抄进 dpo；eval_weights=student |

## 判分要点
- 根因必须落到**评错权重(EMA vs student) + 训练崩溃(配置 bug)**，不能只归因"credit 方法不行"。
- 证据必须来自**student vs EMA 评测对比 + 训练曲线崩溃**。
- 只结论"credit 方法不行" → Fail。