"""
reveal.py  (SOLUTION side — 评测方使用，不要放进 agent 的 workspace)

计算本任务的"根因证据"，并演示修复。核心、稳健、无估计伪影的信号是：
  组内 behavior_version spread —— sync=0，async>0（async 的 GRPO 组混入了多个 policy 版本）。

用法（在 solution/ 下）：
    python reveal.py
"""

import os
from collections import defaultdict

import numpy as np

import diagnostics as diag

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(os.path.dirname(HERE), "workspace", "logs")


def per_group_version_spread(recs):
    g = defaultdict(list)
    for r in recs:
        g[r["group_id"]].append(r["behavior_version"])
    spreads = [float(np.std(v)) for v in g.values() if len(v) > 1]
    frac_mixed = np.mean([1.0 if np.std(v) > 0 else 0.0 for v in g.values() if len(v) > 1])
    return float(np.mean(spreads)), float(frac_mixed)


def mean_reward_by_version(recs, bins=8):
    by_v = defaultdict(list)
    for r in recs:
        by_v[r["behavior_version"]].append(r["reward"])
    vs = sorted(by_v)
    # 按版本分桶看 reward 趋势（聚合跨组，避免单组内单样本伪影）
    arr = [(v, float(np.mean(by_v[v])), len(by_v[v])) for v in vs]
    return arr


def main():
    sync = diag.load_jsonl(os.path.join(LOGS, "sync_grpo.jsonl"))
    asyn = diag.load_jsonl(os.path.join(LOGS, "async_naive.jsonl"))

    print("=" * 60)
    print("根因证据（评测方视角）")
    print("=" * 60)

    s_sp, s_mixed = per_group_version_spread(sync)
    a_sp, a_mixed = per_group_version_spread(asyn)
    print("[主证据] 组内 behavior_version spread（最稳健、无估计伪影）")
    print(f"  sync : 平均 spread={s_sp:.2f}, 含混合版本的组占比={s_mixed:.0%}")
    print(f"  async: 平均 spread={a_sp:.2f}, 含混合版本的组占比={a_mixed:.0%}")
    print("  => async 的每个 GRPO 组都把不同 policy 版本的 rollout 混在一起，")
    print("     group-relative baseline 实际在比较'不同强度策略'生成的 reward。")

    print("\n[标量指标为何没用] normalized advantage variance")
    print(f"  sync={diag.advantage_stats(sync)['adv_variance']:.3f}  "
          f"async={diag.advantage_stats(asyn)['adv_variance']:.3f}  "
          f"（都在~1附近、彼此接近，GRPO 的 /std 归一化把组构成差异洗掉了）")

    print("\n[辅助信号，需谨慎] 版本偏移占比 offset_fraction")
    print(f"  sync={diag.bias_decomposition(sync)['offset_fraction']:.3f}  "
          f"async={diag.bias_decomposition(asyn)['offset_fraction']:.3f}")
    print("  注意：0/1 奖励 + 组内每个版本样本少时，该指标会被'单版本单样本'高估，")
    print("  仅作方向性参考；主证据以 behavior_version spread 为准。")

    print("\n[修复演示] 按 behavior_version 分组再标准化（version-grouped），")
    print("  同组内只比较同版本 rollout，消除跨版本比较 → spread 归 0、baseline 不再混版本。")


if __name__ == "__main__":
    main()
