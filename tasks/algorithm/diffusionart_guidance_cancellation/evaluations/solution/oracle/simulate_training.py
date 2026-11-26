import json,os
HERE=os.path.dirname(os.path.abspath(__file__));TASK=os.path.abspath(os.path.join(HERE,"..","..",".."))
WS=os.path.join(TASK,"workspace");LOGS=os.path.join(WS,"frozen_evidence")
os.makedirs(LOGS,exist_ok=True)
# defective: guidance ρ≈0.024(抵消); reference: restoration ρ≈0.474(存活)
recs=[{"pathway":"guidance","rho":0.024,"survives":False},{"pathway":"restoration","rho":0.474,"survives":True}]
with open(os.path.join(LOGS,"metrics.jsonl"),"w") as f:
  for r in recs:f.write(json.dumps(r)+"\n")
rep="""# 训练报告（待诊断）
扩散模型 reward 微调不提升。
- guidance 通路梯度相干性 ρ≈0.024（几乎全抵消）
- restoration 通路 ρ≈0.474（存活）
问题：为什么 reward 微调不涨？根因是什么？
"""
open(os.path.join(WS,"training_report.txt"),"w").write(rep)
json.dump({"run":"diffusionart","symptom":"reward微调不涨","rho_guid":0.024,"rho_rest":0.474},
  open(os.path.join(WS,"metrics_summary.json"),"w"),ensure_ascii=False,indent=2)
print("rho_guid=0.024(抵消) rho_rest=0.474(存活) -> guidance抵消: True")
