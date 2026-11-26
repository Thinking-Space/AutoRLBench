import json,os
H=os.path.dirname(os.path.abspath(__file__));W=os.path.join(os.path.abspath(os.path.join(H,"..","..","..")),"workspace","frozen_evidence")
os.makedirs(W,exist_ok=True)
recs=[{"step":t,"zero_adv_frac":0.85,"grad_norm":0.001} for t in range(1,21)]
with open(os.path.join(W,"metrics.jsonl"),"w") as f:
 for r in recs:f.write(json.dumps(r)+"\n")
open(os.path.join(os.path.dirname(W),"training_report.txt"),"w").write("# 报告\n训练停滞 grad_norm≈0。根因？\n")
print("advantage全0 -> True")
