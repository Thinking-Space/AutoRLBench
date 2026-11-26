# Solution 指令（修复 + rerun 阈值）

在完成 issue 诊断（根因=优化侧配置：LR 过低 + 掩码过窄 + batch 偏小，叠加数据噪声）的基础上：

## 修复
- 学习率：1.5e-6 → 5e-6~1e-5，配 warmup（数百步）+ cosine decay；
- 掩码率：0.05~0.25 → 接近 0.9（余弦采样），消除训练/推理不一致；
- 有效 batch：grad_accum 16 → 32~64，降低梯度方差；
- 梯度裁剪：0.5 → 1.0（提 LR 后配合）；
- 数据采样：充分 shuffle / 按风格均衡采样。

## Rerun 阈值
修复后全量多风格 T2I 应满足：
- 末段 loss 显著低于修复前（接近 reference 水平）；
- 震荡幅度（末段 std）显著下降；
- 每个风格不被稀释（固定 prompt bank + 固定 seed 评估）。

## 产出
- 修复后的配置；
- rerun 的 loss 曲线与最终指标；
- 修复前后对比（loss 末段均值、震荡 std）。