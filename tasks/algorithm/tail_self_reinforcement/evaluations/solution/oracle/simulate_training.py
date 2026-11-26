import json,os,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));TASK=os.path.abspath(os.path.join(HERE,"..","..",".."))
WS=os.path.join(TASK,"workspace");LOGS=os.path.join(WS,"frozen_evidence")
os.makedirs(LOGS,exist_ok=True)
# defective: 弱层(长任务)成功率随训练下降; reference: coverage-aware稳定
steps=50;recs=[]
for t in range(1,steps+1):
  weak=0.5*0.98**t;strong=0.9
  recs.append({"step":t,"stratum":"weak","success":round(weak,4)})
  recs.append({"step":t,"stratum":"strong","success":round(strong,4)})
with open(os.path.join(LOGS,"metrics.jsonl"),"w") as f:
  for r in recs:f.write(json.dumps(r)+"\n")
rep="""# 训练报告（待诊断）
弱任务层越训越弱，gap 扩大。
- weak 层 success: step1=0.50 → step50=0.18
- strong 层 success: 0.90 稳定
问题：为什么弱层越训越弱？根因是什么？
"""
open(os.path.join(WS,"training_report.txt"),"w").write(rep)
json.dump({"run":"tail_self","symptom":"弱层越训越弱","weak_start":0.5,"weak_end":0.18},
  open(os.path.join(WS,"metrics_summary.json"),"w"),ensure_ascii=False,indent=2)
print("weak 0.50→0.18 strong 0.90 -> 富者愈富: True")
