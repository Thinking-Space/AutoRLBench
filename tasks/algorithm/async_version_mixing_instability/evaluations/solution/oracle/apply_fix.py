#!/usr/bin/env python3
"""
oracle/apply_fix.py —— 修复的参考实现（version-grouped GRPO 标准化演示）。

对一组 rollout：按 behavior_version 分子组，子组内标准化 reward；
展示修复后组内不再混版本 → baseline 不再比较"不同强度策略"的 rollout。
"""
import json
import os
from collections import defaultdict

import numpy as np


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def version_grouped_adv(recs):
    """按 group_id 分大组，大组内再按 behavior_version 分子组，子组内标准化。"""
    groups = defaultdict(list)
    for r in recs:
        groups[r["group_id"]].append(r)
    adv = []
    for gid, rs in groups.items():
        sub = defaultdict(list)
        for r in rs:
            sub[r["behavior_version"]].append(r)
        for v, rs_v in sub.items():
            if len(rs_v) < 2:
                continue
            a = np.array([r["reward"] for r in rs_v], float)
            adv.extend(((a - a.mean()) / (a.std() + 1e-8)).tolist())
    return adv


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    logs = os.path.join(here, "..", "..", "..", "workspace", "frozen_evidence")
    asyn = load(os.path.join(logs, "async_naive.jsonl"))
    sync = load(os.path.join(logs, "sync_grpo.jsonl"))

    def spread(recs):
        g = defaultdict(list)
        for r in recs:
            g[r["group_id"]].append(r["behavior_version"])
        return float(np.mean([np.std(v) for v in g.values() if len(v) > 1]))

    print(f"修复前 组内 version spread  sync={spread(sync):.2f}  async={spread(asyn):.2f}")
    print("修复 = 按 behavior_version 分子组标准化 → 子组内单一版本，spread 归 0，baseline 不再混版本。")
    a = version_grouped_adv(asyn)
    s = version_grouped_adv(sync)
    print(f"修复后 advantage 规模  sync_var={np.var(s):.3f}  async_var={np.var(a):.3f}  （均为子组内标准化，构成一致）")


if __name__ == "__main__":
    main()
