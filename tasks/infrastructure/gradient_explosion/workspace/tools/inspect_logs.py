#!/usr/bin/env python3
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"records={len(rows)} fields={sorted(rows[0].keys())}")
for r in rows[:12]:print(f"  {r}")
