import json,os
H=os.path.dirname(os.path.abspath(__file__));W=os.path.join(os.path.abspath(os.path.join(H,"..","..","..")),"workspace","frozen_evidence")
os.makedirs(W,exist_ok=True)
recs=[{"step":t,"kl":round(0.1*1.5**t,2),"reward":round(max(0,0.8-0.02*t),3)} for t in range(1,21)]
with open(os.path.join(W,"metrics.jsonl"),"w") as f:
 for r in recs:f.write(json.dumps(r)+"\n")
open(os.path.join(os.path.dirname(W),"training_report.txt"),"w").write("# 报告\nKL爆炸 reward崩。根因？\n")
print("KL爆炸 -> True")
