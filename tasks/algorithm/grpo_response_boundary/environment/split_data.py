from pathlib import Path
from sys import argv

import pyarrow as pa
import pyarrow.parquet as pq


INSTRUCTION = (
    "Solve the following AIME problem. Think step by step, explain your reasoning, "
    "and finish with the integer answer in \\boxed{}.\n\n"
)


def training_row(row):
    return {
        "data_source": "aime",
        "prompt": [
            {"role": "system", "content": ""},
            {"role": "user", "content": INSTRUCTION + row["problem"]},
        ],
        "reward_model": {"ground_truth": str(row["answer"])},
        "extra_info": {"index": int(row["id"])},
    }


def main(root):
    source = root / "assets/aime2024/data/train-00000-of-00001.parquet"
    target = root / "assets/split"
    target.mkdir(exist_ok=True)
    rows = sorted(pq.read_table(source).to_pylist(), key=lambda row: int(row["id"]))
    validation, training = rows[:4], rows[4:20]
    for name, split in [("train", training), ("validation", validation)]:
        pq.write_table(pa.Table.from_pylist([training_row(row) for row in split]), target / f"{name}.parquet")


if __name__ == "__main__":
    main(Path(argv[1]))
