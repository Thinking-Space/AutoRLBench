"""
simulate_training.py  (SOLUTION side — 评测方使用，不要放进 agent 的 workspace)

生成一个"异步 GRPO 组内版本混合致训练失稳"的、自洽的可诊断 run。
与旧版不同：这里显式模拟了一个会"学习"的标量 skill θ，使得——
  - sync run 的 θ 平滑上升 → 最终 reward 高、无崩溃；
  - async run 因组内混入旧版本、advantage 被版本偏移污染 → θ 增长被拖慢且偶发回退(崩溃)
    → 最终 reward 明显更低、出现 collapse 事件；
  - 两者的 normalized advantage variance 都≈1(被 /std 洗掉，这是诊断陷阱)；
  - async 组内不同版本的平均 reward 不同 → 版本偏移占比(offset_fraction)高。

产物：
  workspace/logs/sync_grpo.jsonl, workspace/logs/async_naive.jsonl   (给 agent 的原始日志)
  workspace/training_report.txt, workspace/metrics_summary.json       (给 agent 的输入，仅中性+陷阱指标)
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
WS = os.path.join(TASK, "workspace")
LOGS = os.path.join(WS, "logs")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(n_steps=80, n_prompts=24, group_size=8, max_lag=8, seed=0):
    rng = np.random.default_rng(seed)
    difficulty = rng.uniform(0.0, 2.0, size=n_prompts)   # 中等难度，保证 reward 不饱和到 0/1

    # 两条独立的 policy skill 轨迹 theta[version]（都在"学习"，保证组内版本差异能产生 reward 差）
    theta_sync = np.zeros(n_steps + 1)
    theta_async = np.zeros(n_steps + 1)
    gain = 0.022                     # 每步干净学习增益（θ_sync 稳步上升）
    async_collapse_steps = []
    for t in range(1, n_steps + 1):
        theta_sync[t] = theta_sync[t - 1] + gain + rng.normal(0, 0.002)
        # async：版本混合使 advantage 带偏差 -> 抵消部分增益，并偶发中等回退(崩溃)
        bias = 0.012 + rng.normal(0, 0.002)
        shock = 0.0
        if rng.random() < 0.05:      # 偶发崩溃（温和，避免 reward 触底、保持组内非退化）
            shock = rng.uniform(0.06, 0.12)
            async_collapse_steps.append(t)
        theta_async[t] = theta_async[t - 1] + gain - bias - shock

    def gen(path, mode, theta):
        recs = []
        for t in range(1, n_steps + 1):
            for p in range(n_prompts):
                gid = f"g_{p}_{t}"
                if mode == "sync":
                    versions = [t] * group_size
                else:  # async_naive：组内混多个滞后版本
                    versions = [max(1, t - int(rng.integers(0, max_lag + 1))) for _ in range(group_size)]
                for v in versions:
                    # reward 反映"生成该 rollout 的那个版本"的策略水平 theta[v]
                    p_succ = float(sigmoid(theta[v] - difficulty[p]))
                    reward = float(rng.random() < p_succ)   # 0/1 verifier
                    lag = t - v
                    recs.append({
                        "train_step": t, "prompt_id": int(p), "group_id": gid,
                        "reward": reward, "behavior_version": int(v), "current_version": int(t),
                        "mc_return": reward,
                        "kl_to_ref": float(0.01 * t + rng.normal(0, 0.002)),
                    })
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return recs

    os.makedirs(LOGS, exist_ok=True)
    sync = gen(os.path.join(LOGS, "sync_grpo.jsonl"), "sync", theta_sync)
    asyn = gen(os.path.join(LOGS, "async_naive.jsonl"), "async_naive", theta_async)
    return sync, asyn, async_collapse_steps


# ---- 仅用于生成"给 agent 的中性指标"，不含任何根因提示 ----

def _norm_adv_var(recs):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in recs:
        groups[r["group_id"]].append(r["reward"])
    adv = []
    for rs in groups.values():
        a = np.array(rs, float)
        if len(a) < 2:
            continue
        adv.extend(((a - a.mean()) / (a.std() + 1e-8)).tolist())
    return float(np.var(adv)) if adv else 0.0


def _final_mean_reward(recs):
    from collections import defaultdict
    by_step = defaultdict(list)
    for r in recs:
        by_step[r["train_step"]].append(r["reward"])
    last = max(by_step)
    return float(np.mean(by_step[last]))


def _collapse_events(recs, drop=0.3, window=3):
    from collections import defaultdict
    by_step = defaultdict(list)
    for r in recs:
        by_step[r["train_step"]].append(r["reward"])
    steps = sorted(by_step)
    means = [np.mean(by_step[s]) for s in steps]
    n = 0
    for i in range(window, len(means)):
        prev = np.mean(means[i - window:i])
        if prev - means[i] > drop * (abs(prev) + 1e-8):
            n += 1
    return n


def write_agent_facing(sync, asyn):
    nav_s, nav_a = _norm_adv_var(sync), _norm_adv_var(asyn)
    fr_s, fr_a = _final_mean_reward(sync), _final_mean_reward(asyn)
    col_a = _collapse_events(asyn)

    metrics = {
        "run_under_diagnosis": "async_naive",
        "reference_run": "sync_grpo",
        "symptom": "异步 run 收敛到明显更低的 reward 水平（学习效率显著变差），其它常规标量指标看不出异常",
        "agent_visible_metrics": {
            "final_mean_reward_sync": round(fr_s, 4),
            "final_mean_reward_async": round(fr_a, 4),
            "normalized_advantage_variance_sync": round(nav_s, 4),
            "normalized_advantage_variance_async": round(nav_a, 4),
            "lr/batch/data": "与同步参考 run 完全一致，未改动",
        },
        "available_raw_materials": [
            "logs/sync_grpo.jsonl", "logs/async_naive.jsonl",
            "每条 rollout 含字段：train_step, behavior_version, current_version, reward, mc_return, kl_to_ref",
        ],
        "hint": "你可以用 workspace/tools/inspect_logs.py 读取日志、查看可用字段与每步 reward 曲线。",
    }
    with open(os.path.join(WS, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report = f"""# 训练报告（待诊断）

## 设置
- 算法：GRPO，异步 rollout（rollout worker 落后 learner 若干个 policy 版本）
- 参考：同一配置的同步 GRPO run（sync_grpo）
- 任务：数学推理（0/1 verifier）
- 学习率 / batch size / 数据 pipeline：与同步参考 run 完全一致，未改动

## 现象
- 同步 run 最终平均 reward：{fr_s:.3f}
- 异步 run 最终平均 reward：{fr_a:.3f}（明显更低，学习效率显著变差）

## 我检查过、但看不出问题的指标
- normalized advantage variance：同步 {nav_s:.3f} / 异步 {nav_a:.3f}
  —— 两者都在 ~1 附近、彼此接近，advantage 尺度看起来都正常，无法据此区分哪个 run 有问题。
- 没有观察到 loss / reward 的剧烈尖峰或爆炸，训练"看起来"是平稳的。

## 问题
异步 run 为什么会学得更差？请定位根因，并说明你用什么指标 / 证据确认。
原始日志在 logs/ 下，每条 rollout 保留了生成它的 behavior_version。
"""
    with open(os.path.join(WS, "training_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return nav_s, nav_a, fr_s, fr_a, col_a


if __name__ == "__main__":
    sync, asyn, collapses = simulate(seed=0)
    nav_s, nav_a, fr_s, fr_a, col_a = write_agent_facing(sync, asyn)

    # 用 solution 侧 diagnostics 计算"根因证据"（稳健信号），确认自洽
    import diagnostics as diag  # 同目录
    s_recs = diag.load_jsonl(os.path.join(LOGS, "sync_grpo.jsonl"))
    a_recs = diag.load_jsonl(os.path.join(LOGS, "async_naive.jsonl"))
    s_adv, a_adv = diag.advantage_stats(s_recs), diag.advantage_stats(a_recs)
    s_bias, a_bias = diag.bias_decomposition(s_recs), diag.bias_decomposition(a_recs)
    # reward vs lag 相关（稳健根因证据）
    import numpy as _np
    lags = _np.array([r["current_version"] - r["behavior_version"] for r in a_recs], float)
    rews = _np.array([r["reward"] for r in a_recs], float)
    corr = float(_np.corrcoef(lags, rews)[0, 1])

    print("=== 自洽性检查 ===")
    print(f"[失败成立]   final reward  sync={fr_s:.3f}  async={fr_a:.3f}  -> async 更低: {fr_a < fr_s}")
    print(f"[有崩溃]     async collapse events(报告口径)={col_a}  注入崩溃步数={len(collapses)}")
    print(f"[陷阱成立]   normalized adv var  sync={nav_s:.3f}  async={nav_a:.3f}  "
          f"-> 两者接近、都不显著偏离、无法区分: {abs(nav_s-nav_a)<0.15}")
    print("--- 稳健根因证据（给评测方）---")
    print(f"组内 behavior_version spread  sync={s_adv['mean_group_version_spread']:.2f}  "
          f"async={a_adv['mean_group_version_spread']:.2f}  -> 只有 async>0")
    print(f"reward 与 lag 的相关系数(async)={corr:.3f}  -> 负相关=越旧版本 reward 越低")
    print(f"版本偏移占比 offset_fraction  sync={s_bias['offset_fraction']:.3f}  "
          f"async={a_bias['offset_fraction']:.3f}  (0/1 奖励下为小但>0 的辅助信号)")
