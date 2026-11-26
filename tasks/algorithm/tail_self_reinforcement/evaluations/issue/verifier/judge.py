#!/usr/bin/env python3
import json,sys
text=open(sys.argv[1] if len(sys.argv)>1 else "diagnosis.md").read().lower()
rc=sum(1 for k in ['age', '过滤', 'filter', '富者愈富', 'rich', 'coverage'] if k in text)
ev=sum(1 for k in ['gap', '漂移', 'drift', '曝光', 'exposure'] if k in text)
wrong=any(k in text for k in ['探索', 'exploration']) and rc==0
verdict=1.0 if (rc>0 and not wrong) else 0.2
score=0.2*verdict+0.6*min(1,rc/3)+0.2*min(1,ev/3)
if wrong: score*=0.3
print(json.dumps({"score":round(score,3)}));sys.exit(0 if score>=0.8 else 1)
