import json,os
HERE=os.path.dirname(os.path.abspath(__file__));TASK=os.path.abspath(os.path.join(HERE,"..","..",".."))
WS=os.path.join(TASK,"workspace");LOGS=os.path.join(WS,"frozen_evidence")
os.makedirs(LOGS,exist_ok=True)
recs=[{"method":"vanilla","viol_rate":1.0,"mean_cost":165.9},{"method":"cvar","viol_rate":1.0,"mean_cost":166.7},{"method":"safe","viol_rate":0.0,"mean_cost":167.2}]
with open(os.path.join(LOGS,"metrics.jsonl"),"w") as f:
  for r in recs:f.write(json.dumps(r)+"\n")
rep="""# 训练报告（待诊断）
RL 调度 100% 违约。
- vanilla viol_rate=1.0, cvar viol_rate=1.0, safe viol_rate=0.0
问题：为什么 vanilla/cvar 全违约？根因是什么？
"""
open(os.path.join(WS,"training_report.txt"),"w").write(rep)
json.dump({"run":"rl_dispatch","symptom":"100%违约","viol_vanilla":1.0,"viol_safe":0.0},
  open(os.path.join(WS,"metrics_summary.json"),"w"),ensure_ascii=False,indent=2)
print("viol vanilla=1.0 cvar=1.0 safe=0.0 -> 约束未进reward: True")
