import re
import os
import pandas as pd, random, sys
WS = os.environ.get("WS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TPL = """Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command."""
df = pd.read_json(f"{WS}/data/pool_clean.jsonl", lines=True, dtype={"answer": str})
sky = df[df.src.str.startswith("skywork") & df['diff'].between(1/16, 12/16)]
pol = df[(df.src == "polaris") & df['diff'].between(1/8, 6/8)]
dap = df[df.src == "dapo"]
sel = pd.concat([sky, pol, dap]).sample(frac=1.0, random_state=0)
sel = sel[sel.problem.str.len() < 3000]
multi = sel.problem.str.contains(r"\(1\)|\(I\)|\(II\)|\(2\)|Part \(a\)|\(a\)") 
sel = sel[~multi]
print(len(sky), len(pol), len(dap), "->", len(sel))
recs = []
for i, r in enumerate(sel.itertuples()):
    recs.append(dict(data_source="aime_like", prompt=[{"role": "user", "content": TPL.format(prompt=r.problem)}],
                     ability="math", reward_model={"style": "rule", "ground_truth": str(int(float(r.answer))) if re.fullmatch(r"-?\d+(\.0+)?", str(r.answer)) else str(r.answer)},
                     extra_info={"index": i, "src": r.src}))
out = pd.DataFrame(recs)
out.iloc[:-64].to_parquet(f"{WS}/data/rl_train.parquet")
# small val set: 64 held-out training-pool problems (not test data)
out.tail(64).to_parquet(f"{WS}/data/rl_val.parquet")
print(out.iloc[0].prompt[0]['content'])
