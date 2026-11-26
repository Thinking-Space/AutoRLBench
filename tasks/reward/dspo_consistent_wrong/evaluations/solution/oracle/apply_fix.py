import json,os
HERE=os.path.dirname(os.path.abspath(__file__));LOGS=os.path.join(os.path.abspath(os.path.join(HERE,"..","..","..")),"workspace","frozen_evidence")
d=[json.loads(l) for l in open(os.path.join(LOGS,"metrics.jsonl"))]
r=[json.loads(l) for l in open(os.path.join(LOGS,"reference.jsonl"))]
print(f"修复前 coverage={sum(x['correct'] for x in d)/len(d):.3f}")
print(f"修复目标 coverage={sum(x['correct'] for x in r)/len(r):.3f}")
print("修复=verifier正向选择信号→coverage↑→flip wrong→correct。Rerun:coverage>0.3。")
