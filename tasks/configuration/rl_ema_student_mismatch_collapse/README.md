# AutoRLBench Task: rl_ema_student_mismatch_collapse

- **Category**: `configuration`
- **Level**: issue
- **Domain**: RL（GRPO 类）在 VLA/LIBERO-Long 上，三组 credit 变体
- **One-liner**: 三组 RL 全低于 SFT(44%)，但根因不是 credit 方法不行，而是①评错权重(训练用 EMA 却评 student)②训练崩溃(nft_weight_scale 把 mse 形式抄进 dpo 形式→几步内成功率归零自锁)。

## 诊断陷阱
三组 RL 全低于 SFT、A-gae 未超 A-uniform → 容易误判"credit assignment 方法不行"。
实际：评的是 student 权重（训练用 EMA），且 B1 训练几步内崩溃自锁（配置 bug）。

## 根因（oracle 摘要）
1. **EMA/student 评错权重**：训练用 EMA 采样，却评 student → RL 臂看起来全低于 SFT；评 EMA 则 ≥SFT。
2. **训练崩溃**：nft_weight_scale 把 mse 形式 [1.0, 0.2] 抄进 dpo 形式 → 能量对称违反 → 几步内成功率归零、自锁。

## 构造
`evaluations/solution/oracle/simulate_training.py` 生成 frozen_evidence（模拟 run，自洽可复现）。