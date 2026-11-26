# AutoRLBench Task: diffusionart_guidance_cancellation

- **Category**: `algorithm`
- **Level**: issue
- **One-liner**: 扩散 RL reward 微调不涨，根因是 guidance 梯度几乎全抵消(ρ_guid≈0.024)，restoration 存活(ρ_rest≈0.474)

## 诊断陷阱
以为学习率/步数不够，实际是 guidance 梯度被逐状态 Jacobian 门控抵消

## 根因（oracle 摘要）
guidance 梯度 ρ_guid≈0.024 几乎全抵消；restoration 存活 ρ_rest≈0.474；需 NTK 预条件 restoration

## 构造
`evaluations/solution/oracle/simulate_training.py` 生成 frozen_evidence（模拟 run，自洽可复现）。
