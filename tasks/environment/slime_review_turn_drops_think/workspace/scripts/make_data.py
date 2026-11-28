"""Build the RL prompt set: MATH (hendrycks, lighteval copy) train split, level 5,
one JSON line per problem: {"prompt": problem, "label": reference answer from the last \\boxed{}}."""
import argparse, json
import pandas as pd
from slime.rollout.rm_hub.math_utils import extract_boxed_answer

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="MATH-lighteval data/train-00000-of-00001.parquet")
ap.add_argument("--out", required=True)
a = ap.parse_args()

df = pd.read_parquet(a.src)
df = df[df.level == "Level 5"]
n = 0
with open(a.out, "w") as f:
    for r in df.itertuples():
        ans = extract_boxed_answer(r.solution)
        if not ans or len(ans) > 40:
            continue
        f.write(json.dumps({"prompt": r.problem, "label": ans, "level": r.level, "type": r.type}) + "\n")
        n += 1
print("wrote", n)
