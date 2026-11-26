import json,os,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));TASK=os.path.abspath(os.path.join(HERE,"..","..",".."))
WS=os.path.join(TASK,"workspace");LOGS=os.path.join(WS,"frozen_evidence")
os.makedirs(LOGS,exist_ok=True);rng=np.random.default_rng(0)
G=8;n_groups=200
# defective: 整组都错(coverage≈0), became_correct≈0
def gen(coverage,flip,seed):
  rng=np.random.default_rng(seed);recs=[]
  for g in range(n_groups):
    has_correct=rng.random()<coverage
    for i in range(G):
      correct=has_correct and i==0
      recs.append({"group":g,"rollout":i,"reward":1.0 if correct else 0.0,"correct":correct})
  return recs
def write(p,recs):
  with open(p,"w") as f:
    for r in recs:f.write(json.dumps(r)+"\n")
write(os.path.join(LOGS,"metrics.jsonl"),gen(0.02,0.0,1))
write(os.path.join(LOGS,"reference.jsonl"),gen(0.35,0.4,2))
cov_def=0.02;cov_ref=0.35
rep=f"""# 训练报告（待诊断）
consensus reward 训练不涨，became_correct≈0。
- 当前 group coverage(含正确答案的组占比)≈{cov_def}
- 对照(verifier)coverage≈{cov_ref}
问题：为什么 consensus reward 翻不了盘？根因是什么？
"""
open(os.path.join(WS,"training_report.txt"),"w").write(rep)
json.dump({"run":"dspo","symptom":"became_correct≈0","coverage_defective":cov_def,"coverage_reference":cov_ref},
  open(os.path.join(WS,"metrics_summary.json"),"w"),ensure_ascii=False,indent=2)
print(f"coverage defective={cov_def} reference={cov_ref} -> 结构问题: {cov_def<0.1}")
