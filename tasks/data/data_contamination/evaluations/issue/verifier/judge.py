#!/usr/bin/env python3
import json,sys
text=open(sys.argv[1] if len(sys.argv)>1 else "d.md").read().lower()
rc=sum(1 for k in ['污染', 'contamination', '重叠', 'overlap', '泄漏', 'leak', 'dedup'] if k in text)
ev=sum(1 for k in ['eval', '评测'] if k in text)
verdict=1.0 if rc>0 else 0.2
score=0.2*verdict+0.6*min(1,rc/3)+0.2*min(1,ev/3)
print(json.dumps({"score":round(score,3)}));sys.exit(0 if score>=0.8 else 1)
