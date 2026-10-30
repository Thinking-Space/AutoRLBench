"""Build data/math500/test.parquet (MATH-500 as a validation set) in the schema of verl's math_dataset.py.

Same instruction suffix as MATH-lighteval, data_source "HuggingFaceH4/MATH-500", ground truth = the dataset's
`answer` field. Written with `datasets` so string columns match verl's own preprocessors (verl concatenates the
validation files with datasets, which refuses to align `string` with pandas' `large_string`).
Usage: python prepare_math500.py <raw MATH-500 dir> <output dir>"""
import os
import sys

import datasets

INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

raw, out = sys.argv[1], sys.argv[2]
rows = [
    {
        "data_source": "HuggingFaceH4/MATH-500",
        "prompt": [{"role": "user", "content": ex["problem"] + " " + INSTRUCTION}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ex["answer"]},
        "extra_info": {"split": "test", "index": i},
    }
    for i, ex in enumerate(datasets.load_dataset(raw)["test"])
]
os.makedirs(out, exist_ok=True)
datasets.Dataset.from_list(rows).to_parquet(os.path.join(out, "test.parquet"))
print(f"wrote {len(rows)} rows to {out}/test.parquet")
