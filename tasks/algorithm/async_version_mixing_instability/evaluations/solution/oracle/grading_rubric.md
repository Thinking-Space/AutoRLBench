# 评分标准（Grading Rubric，评测用）

对被测 agent 的诊断按 AutoRLBench 四档评分：

## Pass（通过）
Agent 指出根因是**异步导致同一 GRPO 组内混入了多个 policy 版本，使 group-relative baseline
在比较不同强度策略生成的 rollout，从而污染 advantage / credit assignment**，并给出稳健证据：
- 按 group 统计 `behavior_version` 的构成 / spread（sync=0 vs async>0，绝大多数组混版本）；
- 或等价地指出"组里 rollout 来自不同 policy 版本，mean 基线不再对应单一策略"。
（能同时识破 normalized advantage variance 无法区分、是加分项。）

## Weak evidence（弱证据）
方向对（提到版本陈旧 / off-policy / 组内策略不一致），但：
- 没有用组构成 / 版本 spread 等可验证证据，只是定性猜测；或
- 只泛泛说"advantage 有偏/off-policy"，没具体到"组内混版本污染 group baseline"。

## Wrong cause（错因）
归因到与根因无关的方向，例如：学习率过大、数据质量/污染、reward hacking、
batch size、KL 太大——且未触及"组内版本混合污染 group baseline"。
（本 run 的 lr/数据/batch 与同步参考 run 完全一致，是干扰项。）

## Fail（失败）
被"normalized advantage variance≈1、训练看起来平稳"误导，得出"训练正常/没问题"，
或无法给出根因。

## 关键判别点
这道题的区分度在于：agent 能否意识到**常规标量指标（含 normalized advantage variance）
无法暴露问题，必须去检查 GRPO 组的"版本构成"**。查组构成、发现混版本 → Pass；
只停在标量指标上 → Fail。
