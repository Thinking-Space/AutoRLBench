#!/usr/bin/env python3
"""中性：读按 horizon 的 success_rate + 分布，不给诊断结论。"""
import json
import sys

rows = []
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"records={len(rows)} fields={sorted(rows[0].keys())}")
print("success_rate by horizon:")
for r in rows:
    bar = "#" * int(r["success_rate"] * 40)
    print(f"  h={r['horizon']:>2}  {r['success_rate']:.3f}  {bar}")