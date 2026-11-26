#!/usr/bin/env python3
import json,sys
text=open(sys.argv[1] if len(sys.argv)>1 else "d.md").read().lower()
rc=sum(1 for k in ['normalization', '归一化', 'length', '长度', 'bias'] if k in text)
ev=sum(1 for k in ['accuracy', 'correct'] if k in text)
verdict=1.0 if rc>0 else 0.2
score=0.2*verdict+0.6*min(1,rc/3)+0.2*min(1,ev/3)
print(json.dumps({"score":round(score,3)}));sys.exit(0 if score>=0.8 else 1)
