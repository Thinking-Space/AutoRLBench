#!/usr/bin/env python3
"""
workspace/tools/inspect_logs.py —— 中性：读 loss 曲线 + 配置，不给任何诊断结论。
用法：python tools/inspect_logs.py frozen_evidence/loss_full_defective.jsonl
"""
import json
import sys

rows = []
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"records={len(rows)} fields={sorted(rows[0].keys())}")
print("loss curve (every 10 steps):")
for r in rows[::10]:
    bar = "#" * int(r["loss"] * 40)
    print(f"  step {r['step']:>3}  {r['loss']:.3f}  {bar}")
print(f"final-20 mean: {sum(r['loss'] for r in rows[-20:])/20:.3f}")