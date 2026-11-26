import json,os
H=os.path.dirname(os.path.abspath(__file__));W=os.path.join(os.path.abspath(os.path.join(H,"..","..","..")),"workspace","frozen_evidence")
os.makedirs(W,exist_ok=True)
recs=[{"step":t,"reward":round(0.3+0.005*t,3),"accuracy":0.45,"length":round(100+20*t)} for t in range(1,51)]
with open(os.path.join(W,"metrics.jsonl"),"w") as f:
 for r in recs:f.write(json.dumps(r)+"\n")
open(os.path.join(os.path.dirname(W),"training_report.txt"),"w").write("# 报告\nreward 涨但 accuracy 不涨(0.45)，length 从100→1100。根因？\n")
print("reward涨 accuracy不涨 length暴涨 -> hacking: True")
