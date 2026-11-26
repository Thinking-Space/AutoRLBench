import json,os
H=os.path.dirname(os.path.abspath(__file__));W=os.path.join(os.path.abspath(os.path.join(H,"..","..","..")),"workspace","frozen_evidence")
os.makedirs(W,exist_ok=True)
recs=[{"step":t,"proxy":round(0.5+0.01*t,3),"true":round(0.5+0.02*t-0.001*t*t,3)} for t in range(1,51)]
with open(os.path.join(W,"metrics.jsonl"),"w") as f:
 for r in recs:f.write(json.dumps(r)+"\n")
open(os.path.join(os.path.dirname(W),"training_report.txt"),"w").write("# 报告\nproxy涨true跌。根因？\n")
print("reward过优化 -> True")
