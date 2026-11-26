"""
inspect_logs.py  (workspace/tools — 给被测 agent 的中性工具，不含任何根因提示)

读取 rollout 日志，打印：每个 run 的字段清单、每步平均 reward 曲线、基础统计。
不计算任何诊断性结论——怎么用这些原始字段去定位问题，由 agent 自己决定。

用法（在 workspace/ 下）：
    python tools/inspect_logs.py logs/async_naive.jsonl
    python tools/inspect_logs.py logs/sync_grpo.jsonl
"""

import json
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    if len(sys.argv) < 2:
        print("usage: python tools/inspect_logs.py <path-to.jsonl>")
        return
    rows = load(sys.argv[1])
    print(f"# {sys.argv[1]}")
    print(f"records: {len(rows)}")
    print(f"fields : {sorted(rows[0].keys())}")

    by_step = defaultdict(list)
    for r in rows:
        by_step[r["train_step"]].append(r["reward"])
    steps = sorted(by_step)
    print(f"train_steps: {steps[0]}..{steps[-1]}  ({len(steps)} steps)")
    print("\nmean reward per step:")
    for s in steps:
        m = sum(by_step[s]) / len(by_step[s])
        bar = "#" * int(m * 40)
        print(f"  step {s:>3}  {m:4.2f}  {bar}")

    n_groups = len(set(r["group_id"] for r in rows))
    print(f"\ngroups: {n_groups}, rollouts/group ~ {len(rows)/max(n_groups,1):.1f}")
    print("（提示：原始字段都在上面，具体要看什么、怎么归因，由你决定。）")


if __name__ == "__main__":
    main()
