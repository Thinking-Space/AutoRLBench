# 根因与修复（Ground Truth —— 评测用，不给被测 agent）

## 任务
`async-version-mixing-instability` · 类别：**Algorithm** · 域：async GRPO RLVR（数学推理，0/1 verifier）

## 症状（agent 看到的）
异步 run 收敛到明显更低的 reward（本 run：async 0.354 vs sync 0.661），
其它常规标量指标都看不出异常，训练"看起来"是平稳的。

## 真正的根因
异步 rollout 下，同一个 prompt 的一组 G 条轨迹由**不同 policy 版本**生成
（`behavior_version` 各不相同；本 run async 组内平均版本 spread≈2.2，**98% 的组都混了版本**，
而 sync 组 spread=0）。GRPO 的组内标准化 `A_i=(r_i-mean)/std` 隐含假设组内同策略；
版本混合后，组里混进了**不同强度策略**生成的 reward，`mean` 不再是单一策略的条件基线，
advantage 因此同时混入了"动作质量"和"版本强弱差异"，credit assignment 被污染 → 学习效率变差。

## 主证据（最稳健、agent 应给出）
**按 group 统计 `behavior_version` 的构成**：
- sync：每组单一版本，spread=0；
- async：每组混多个版本，spread>0（≈2.2，98% 组混版本）。
→ 直接说明 GRPO 的 group baseline 在比较"不同策略版本"生成的 rollout。

## 诊断陷阱（为什么能卡住 agent）
- **normalized advantage variance 没用**：sync 0.947 / async 0.897，两者都在 ~1 附近且接近
  （GRPO 的 `/std` 归一化把组构成差异洗掉了），无法据此区分或定位。
- **没有 loss/reward 爆炸尖峰**，常规 dashboard 看起来平稳 → 容易误判"没问题"或归因到 lr/数据。
- **简单的 reward-vs-lag 相关性也接近 0**（本 run≈-0.01），naive 地查"reward 是否随 lag 下降"会误导；
  真正被破坏的是 **baseline 的构成**，不是一个简单的 reward-lag 趋势。
- 必须去看**组的构成（版本 spread）**，而标准指标面板不会展示这个。

## 辅助信号（需谨慎，不要作为唯一依据）
版本偏移占比 offset_fraction（sync 0.000 / async 0.611）方向正确，但在 **0/1 奖励 + 组内每版本
样本很少**时会被"单版本单样本"高估，仅作方向性参考；**主证据以 behavior_version spread 为准**。

## 修复（Solution 级别）
- **按 behavior-policy version 分组**后再求 mean/std（version-grouped GRPO），
  同组内只比较同版本 rollout，禁止跨版本标准化；
- 或对过旧轨迹按 age/KL 降权、或丢弃超过 staleness 阈值的轨迹。
- **Rerun 阈值**：修复后 async 的组内版本 spread 应降到 0（或仅在允许的小 staleness 窗口内），
  且最终平均 reward 恢复到接近同步参考 run（差距进入对照容差）。

## 复现（solution/ 下）
```
python simulate_training.py   # 生成 workspace（含给 agent 的 report/metrics/logs）
python reveal.py              # 打印根因证据：组内 version spread 等
```
