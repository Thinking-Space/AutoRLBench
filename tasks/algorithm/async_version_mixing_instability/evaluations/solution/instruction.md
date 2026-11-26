# Solution 指令（修复 + rerun 阈值）

在完成 issue 诊断（组内混多 policy 版本污染 GRPO group baseline）的基础上：

## 修复
实现 version-grouped GRPO 标准化：
- 按 `behavior_version` 分组后，再在组内求 mean/std 构造 group-relative advantage；
- 禁止跨版本标准化；对过旧轨迹按 age/KL 降权或丢弃超过 staleness 阈值的轨迹。

## Rerun 阈值
修复后，异步 run 应满足：
- 组内 behavior_version spread → 0（或仅允许在设定的小 staleness 窗口内）；
- 最终平均 reward 恢复到接近同步参考 run（差距进入对照容差）。

## 产出
- 修复代码；
- 用修复后的设定 rerun 的日志与最终指标；
- 一段结论：修复后与修复前的 reward / 版本 spread 对比。
