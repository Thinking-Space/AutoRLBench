# async_version_mixing_instability

- **Category**: `algorithm`
- **Level**: issue（可升级 solution）
- **Domain**: asynchronous GRPO RLVR（数学推理，0/1 verifier）
- **One-liner**: 异步 rollout 使同一 GRPO 组内混入多个 policy 版本，group-relative baseline 在比较
  "不同强度策略"生成的 rollout，污染 advantage → 异步 run 学得明显更差；而 normalized advantage
  variance 等常规标量指标完全看不出异常。

> 本文件夹为**预排版的原始素材**，供 `skills/package-task` 打包成正式 task。
> `task.toml` / verifier / README 已按官方规范给了合理默认值，**上船后请让 package-task
> 对照 `skills/package-task/reference.md` 核准后再 `python3 -m autorlbench validate tasks/...`**。

## 现象（agent 可见）
- 异步 run 最终 reward 0.354 vs 同步 0.661（明显更差）
- normalized advantage variance 0.897 vs 0.947 —— 两者都≈1、彼此接近，常规标量指标看不出异常
- 训练无爆炸尖峰；reward-vs-lag 相关性≈0 会误导

## 根因（oracle 摘要，详见 evaluations/issue/oracle.md）
异步 rollout 导致每个 GRPO 组混入多个 policy 版本（组内 behavior_version spread≈2.2、98% 组混版本；
同步=0），group baseline 不再是单一策略的条件基线。

## 构造
`workspace/launch_scripts/simulate_training.py` 生成 frozen_evidence 中的日志；
`evaluations/solution/oracle/reveal.py` 打印根因证据。
（当前为模拟 run；如需正式 task，建议以真实小模型 async GRPO run 替换 frozen_evidence。）
