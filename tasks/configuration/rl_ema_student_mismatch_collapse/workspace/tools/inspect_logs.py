#!/usr/bin/env python3
"""中性：读评测结果/训练曲线/配置，不给诊断结论。"""
import json
import sys

rows = []
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
print(f"records={len(rows)} fields={sorted(rows[0].keys())}")
for r in rows[:10]:
    print(f"  {r}")
if len(rows) > 10:
    print(f"  ... ({len(rows)-10} more)")