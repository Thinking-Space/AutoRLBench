import json,os
H=os.path.dirname(os.path.abspath(__file__));W=os.path.join(os.path.abspath(os.path.join(H,"..","..","..")),"workspace","frozen_evidence")
os.makedirs(W,exist_ok=True)
recs=[{"step":t,"avg_length":round(100+15*t),"accuracy":0.45} for t in range(1,51)]
with open(os.path.join(W,"metrics.jsonl"),"w") as f:
 for r in recs:f.write(json.dumps(r)+"\n")
open(os.path.join(os.path.dirname(W),"training_report.txt"),"w").write("# 报告\n输出越来越长但acc不涨。根因？\n")
print("长度偏置 -> True")
