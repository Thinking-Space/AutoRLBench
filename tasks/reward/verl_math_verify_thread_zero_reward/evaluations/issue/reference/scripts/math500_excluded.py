"""Which MATH-500 problems are missing from the validation dumps, and why.

verl's RLHFDataset with filter_overlong_prompts=True drops prompts whose chat-templated token length exceeds
data.max_prompt_length (512). This recomputes that length with the run's tokenizer for all 500 problems, lists the
ones over the limit, and checks against the dump (by problem text) that exactly those are absent.
Usage: python math500_excluded.py <val dump .jsonl> <math500 parquet> <raw MATH-500 test.jsonl> <model dir>"""
import json
import sys

import pandas as pd
from transformers import AutoTokenizer

dump, parquet, raw, model = sys.argv[1:5]
inputs = [json.loads(line)["input"] for line in open(dump)]
math_inputs = [s for s in inputs if '"####"' not in s]
df = pd.read_parquet(parquet)
raw_rows = [json.loads(line) for line in open(raw)]
tok = AutoTokenizer.from_pretrained(model)
over, absent = [], []
for i, row in df.iterrows():
    n = len(tok.apply_chat_template(list(row["prompt"]), add_generation_prompt=True, tokenize=True))
    problem = raw_rows[i]["problem"]
    present = any(problem in s for s in math_inputs)
    if n > 512:
        over.append((raw_rows[i]["unique_id"], n, present))
    elif not present:
        absent.append((raw_rows[i]["unique_id"], n))
print(f"MATH-500 rows: {len(df)}; MATH prompts in the dump: {len(math_inputs)}")
for uid, n, present in over:
    print(f"over 512 tokens: {uid} ({n} tokens), present in dump: {present}")
print(f"under the limit but absent from the dump: {absent}")
