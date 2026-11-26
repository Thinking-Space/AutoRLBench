# Solution 指令（修复 + rerun 阈值）

在完成 issue 诊断（age 门控按 wall-time 过滤长任务 → 长程能力隐形欠训练）的基础上：

## 修复
- 放宽 age 门控（age_limit 4→12），或改用 coverage-aware 调度 + bounded regeneration，
  让长任务也能被准入、贡献梯度；
- 或按 horizon 分层采样，保证长程段有足够曝光。

## Rerun 阈值
修复后应满足：
- 长程(h>10)成功率从 ≈0.003 回升到接近放宽门控水平（≈0.87）；
- 聚合成功率不下降；
- 按 horizon 分层看，长程段不再塌陷。

## 产出
- 修复后的门控/调度配置；
- rerun 的按-horizon 成功率；
- 修复前后对比（长程段成功率）。