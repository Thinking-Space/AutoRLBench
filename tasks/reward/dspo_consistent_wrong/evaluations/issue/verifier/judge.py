#!/usr/bin/env python3
import json,sys
text=open(sys.argv[1] if len(sys.argv)>1 else "diagnosis.md").read().lower()
rc=sum(1 for k in ['coverage', '组内', 'group', '正确答案', '结构'] if k in text)
ev=sum(1 for k in ['coverage', 'group', '组'] if k in text)
wrong=any(k in text for k in ['调参', 'tune', 'learning rate']) and rc==0
verdict=1.0 if (rc>0 and not wrong) else 0.2
score=0.2*verdict+0.6*min(1,rc/3)+0.2*min(1,ev/3)
if wrong: score*=0.3
print(json.dumps({"score":round(score,3)}));sys.exit(0 if score>=0.8 else 1)
