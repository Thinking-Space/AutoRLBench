#!/usr/bin/env python3
import json,sys
text=open(sys.argv[1] if len(sys.argv)>1 else "diagnosis.md").read().lower()
rc=sum(1 for k in ['重要性', 'importance', '连乘', 'product', '指数', 'exponential', 'horizon'] if k in text)
ev=sum(1 for k in ['方差', 'variance', '有效样本', 'sample'] if k in text)
wrong=any(k in text for k in ['学习率', 'lr']) and rc==0
verdict=1.0 if (rc>0 and not wrong) else 0.2
score=0.2*verdict+0.6*min(1,rc/3)+0.2*min(1,ev/3)
if wrong: score*=0.3
print(json.dumps({"score":round(score,3)}));sys.exit(0 if score>=0.8 else 1)
