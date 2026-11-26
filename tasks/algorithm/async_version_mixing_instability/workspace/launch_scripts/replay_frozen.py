#!/usr/bin/env python3
"""
workspace/launch_scripts/replay_frozen.py —— 中性：重放冻结证据、确认可读，不含任何诊断结论。

用法：python launch_scripts/replay_frozen.py <path-to.jsonl>
（agent 可自由使用；不提供任何根因提示。）
"""
import json
import sys
from collections import defaultdict

rows = []
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"records={len(rows)} fields={sorted(rows[0].keys())}")
by_step = defaultdict(list)
for r in rows:
    by_step[r["train_step"]].append(r["reward"])
steps = sorted(by_step)
print(f"mean reward first/last: {sum(by_step[steps[0]])/len(by_step[steps[0]]):.3f} / "
      f"{sum(by_step[steps[-1]])/len(by_step[steps[-1]]):.3f}")
