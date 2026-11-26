import json,os,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));TASK=os.path.abspath(os.path.join(HERE,"..","..",".."))
WS=os.path.join(TASK,"workspace");LOGS=os.path.join(WS,"frozen_evidence")
os.makedirs(LOGS,exist_ok=True)
# defective: 方差随horizon指数爆炸; reference: capping后bound
recs=[]
for h in range(1,21):
  var_uncapped=0.1*1.8**h;var_capped=0.1*min(h,5)
  recs.append({"horizon":h,"var_uncapped":round(var_uncapped,2),"var_capped":round(var_capped,2)})
with open(os.path.join(LOGS,"metrics.jsonl"),"w") as f:
  for r in recs:f.write(json.dumps(r)+"\n")
rep="""# 训练报告（待诊断）
多步 RL 训练不稳、方差爆炸。
- horizon=20: var_uncapped≈1340, var_capped≈0.5
问题：为什么方差爆炸？根因是什么？
"""
open(os.path.join(WS,"training_report.txt"),"w").write(rep)
json.dump({"run":"lag_critic","symptom":"方差爆炸","var_h20_uncapped":1340,"var_h20_capped":0.5},
  open(os.path.join(WS,"metrics_summary.json"),"w"),ensure_ascii=False,indent=2)
print("var h20 uncapped=1340 capped=0.5 -> 指数爆炸: True")
