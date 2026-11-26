"""
diagnostics.py —— 框架无关的核心指标计算模块。

操作对象是 rollout_log_schema.md 定义的 JSONL 日志。所有阶段 A / B 的分析脚本都
调用本模块，你不需要改动这里。

提供：
  - load_jsonl / group_records：读日志、按 (train_step, group_id) 分组
  - advantage_stats：GRPO advantage 及其方差、版本 spread（阶段 A 核心）
  - bias_decomposition：Prop 1 的组内-advantage / 版本偏移分解（阶段 B 核心）
  - calibration_vs_lag：critic calibration error 随 lag 的变化（阶段 B 核心）
  - detect_collapse：训练崩溃/发散检测
  - kl_drift_curve：KL drift 随 step 的曲线
"""

import json
from collections import defaultdict

import numpy as np


def load_jsonl(path):
    """读取 JSONL，返回 list[dict]。空行跳过。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _policy_lag(rec):
    """优先用 logprob_drift（精确 lag），否则用版本号差。"""
    if rec.get("logprob_drift") is not None:
        return float(rec["logprob_drift"])
    return float(rec.get("current_version", 0) - rec.get("behavior_version", 0))


def group_records(records):
    """按 group_id 分组，返回 {group_id: [rec, ...]}。"""
    groups = defaultdict(list)
    for r in records:
        groups[r["group_id"]].append(r)
    return groups


def advantage_stats(records):
    """
    计算 GRPO advantage 及诊断量（阶段 A）。
    返回 dict：
      - adv_variance: 全体 advantage 的方差（越大越不稳）
      - mean_group_version_spread: 组内 behavior_version 的平均 std（衡量组内版本混合程度）
      - mean_policy_lag: 平均 policy lag
      - n_groups, n_records
    """
    groups = group_records(records)
    all_adv = []
    all_adv_raw = []
    version_spreads = []
    lags = []
    for gid, recs in groups.items():
        rewards = np.array([r["reward"] for r in recs], dtype=float)
        if len(rewards) < 2:
            continue
        mu = rewards.mean()
        sd = rewards.std() + 1e-8
        adv = (rewards - mu) / sd
        all_adv.extend(adv.tolist())
        all_adv_raw.extend((rewards - mu).tolist())  # 未除以 std 的原始中心化 advantage
        versions = np.array([r.get("behavior_version", 0) for r in recs], dtype=float)
        version_spreads.append(float(versions.std()))
        lags.extend([_policy_lag(r) for r in recs])

    all_adv = np.array(all_adv) if all_adv else np.array([0.0])
    all_adv_raw = np.array(all_adv_raw) if all_adv_raw else np.array([0.0])
    return {
        # 注意：GRPO 除以 group_std 后 adv_variance 恒≈1，对版本混合不敏感；
        # 判断失稳应优先看 raw_adv_variance 和 bias_decomposition 的 offset_fraction。
        "adv_variance": float(np.var(all_adv)),
        "raw_adv_variance": float(np.var(all_adv_raw)),
        "adv_mean_abs": float(np.mean(np.abs(all_adv))),
        "mean_group_version_spread": float(np.mean(version_spreads)) if version_spreads else 0.0,
        "mean_policy_lag": float(np.mean(lags)) if lags else 0.0,
        "n_groups": len(groups),
        "n_records": len(records),
    }


def bias_decomposition(records):
    """
    Prop 1 的经验分解（阶段 B）。用组内、按 behavior_version 再分子组估计 mu_v，
    分解 (reward - group_mean) = (reward - mu_v) + (mu_v - group_mean)。
    返回：
      - within_version_var: 组内真实 advantage 的方差
      - version_offset_var: 版本偏移 b_v 的方差（Prop 1 声称这一项抬高梯度方差）
      - offset_fraction: 版本偏移方差占总中心化方差的比例（越大说明版本混合污染越严重）
    """
    groups = group_records(records)
    within_all, offset_all, centered_all = [], [], []
    for gid, recs in groups.items():
        rewards = np.array([r["reward"] for r in recs], dtype=float)
        versions = np.array([r.get("behavior_version", 0) for r in recs])
        if len(rewards) < 2:
            continue
        group_mean = rewards.mean()
        # 每条轨迹的 mu_v：同组内同版本 reward 的均值
        mu_v = np.zeros_like(rewards)
        for v in np.unique(versions):
            m = versions == v
            mu_v[m] = rewards[m].mean()
        within = rewards - mu_v
        offset = mu_v - group_mean
        centered = rewards - group_mean
        within_all.extend(within.tolist())
        offset_all.extend(offset.tolist())
        centered_all.extend(centered.tolist())

    within_var = float(np.var(within_all)) if within_all else 0.0
    offset_var = float(np.var(offset_all)) if offset_all else 0.0
    centered_var = float(np.var(centered_all)) if centered_all else 1e-8
    return {
        "within_version_var": within_var,
        "version_offset_var": offset_var,
        "centered_total_var": centered_var,
        "offset_fraction": offset_var / (centered_var + 1e-8),
    }


def calibration_vs_lag(records, n_bins=6):
    """
    critic calibration error 随 lag 的变化（阶段 B）。
    需要 critic_pred 和 mc_return。返回每个 lag bin 的平均 |critic_pred - mc_return|。
    若数据缺失返回 {"available": False}。
    """
    xs, ys = [], []
    for r in records:
        cp = r.get("critic_pred")
        mc = r.get("mc_return")
        if cp is None or mc is None:
            continue
        # critic_pred 可能是 list（step-level），取均值处理
        cp_val = float(np.mean(cp)) if isinstance(cp, (list, tuple)) else float(cp)
        xs.append(_policy_lag(r))
        ys.append(abs(cp_val - float(mc)))
    if len(xs) < n_bins:
        return {"available": False, "reason": "缺少 critic_pred/mc_return 或样本太少"}

    xs = np.array(xs)
    ys = np.array(ys)
    bins = np.linspace(xs.min(), xs.max(), n_bins + 1)
    idx = np.clip(np.digitize(xs, bins) - 1, 0, n_bins - 1)
    per_bin = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        per_bin.append({
            "lag_range": [float(bins[b]), float(bins[b + 1])],
            "n": int(m.sum()),
            "mean_calib_error": float(ys[m].mean()),
        })
    # 计算 calibration error 与 lag 的相关系数（>0 说明 lag 越大 critic 越差）
    corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else 0.0
    return {
        "available": True,
        "per_bin": per_bin,
        "corr_lag_vs_error": corr,
        "increasing_in_lag": bool(corr > 0.1),
    }


def detect_collapse(records, drop_threshold=0.3, window=3):
    """
    简单崩溃检测：按 train_step 聚合平均 reward，检测连续下降超过阈值的事件。
    返回崩溃事件数和发生的 step。
    """
    by_step = defaultdict(list)
    for r in records:
        by_step[r["train_step"]].append(r["reward"])
    steps = sorted(by_step.keys())
    means = [np.mean(by_step[s]) for s in steps]
    events = []
    for i in range(window, len(means)):
        prev = np.mean(means[i - window:i])
        if prev - means[i] > drop_threshold * (abs(prev) + 1e-8):
            events.append(int(steps[i]))
    return {
        "n_collapse_events": len(events),
        "collapse_steps": events,
        "final_mean_reward": float(means[-1]) if means else 0.0,
        "max_mean_reward": float(np.max(means)) if means else 0.0,
    }


def kl_drift_curve(records):
    """按 train_step 聚合 kl_to_ref，返回 (steps, kl)。缺失则返回空。"""
    by_step = defaultdict(list)
    for r in records:
        if r.get("kl_to_ref") is not None:
            by_step[r["train_step"]].append(r["kl_to_ref"])
    steps = sorted(by_step.keys())
    return {
        "available": len(steps) > 0,
        "steps": steps,
        "kl": [float(np.mean(by_step[s])) for s in steps],
    }
