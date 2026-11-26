import json,os
H=os.path.dirname(os.path.abspath(__file__));W=os.path.join(os.path.abspath(os.path.join(H,"..","..","..")),"workspace","frozen_evidence")
os.makedirs(W,exist_ok=True)
recs=[{"step":t,"ref_version":0,"current_version":t,"kl_penalty":0.001} for t in range(1,101)]
with open(os.path.join(W,"metrics.jsonl"),"w") as f:
 for r in recs:f.write(json.dumps(r)+"\n")
open(os.path.join(os.path.dirname(W),"training_report.txt"),"w").write("# 报告\nKL惩罚不生效policy漂移。根因？\n")
print("reference stale -> True")
