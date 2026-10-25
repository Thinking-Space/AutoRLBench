"""Convert the pinned original GSM8K splits to verl RL parquet, without synthetic rows."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from verl.utils.reward_score.gsm8k import compute_score, extract_solution


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(frame, split):
    result = []
    for i, row in frame.iterrows():
        answer = extract_solution(row["answer"], method="strict")
        assert answer is not None
        result.append(dict(
            data_source="openai/gsm8k",
            prompt=[dict(role="user", content=row["question"] + " Let's think step by step and output the final answer after \"####\".")],
            ability="math", reward_model=dict(style="rule", ground_truth=answer),
            extra_info=dict(split=split, index=int(i), question=row["question"], answer=row["answer"]),
        ))
    return pd.DataFrame(result)


def main():
    root = Path(os.environ["W"])
    raw = root / "data/raw/gsm8k/main"
    target = root / "data/gsm8k"
    target.mkdir(exist_ok=True)
    train_raw = pd.read_parquet(raw / "train-00000-of-00001.parquet")
    test_raw = pd.read_parquet(raw / "test-00000-of-00001.parquet")
    assert len(train_raw) == 7473 and len(test_raw) == 1319
    train = convert(train_raw, "train")
    test = convert(test_raw, "test")
    # Only official test examples are eligible for the small periodic validation set.
    indices = sorted(np.random.default_rng(2026).choice(len(test), 64, replace=False).tolist())
    val = test.iloc[indices].reset_index(drop=True)
    train.to_parquet(target / "train.parquet", index=False)
    test.to_parquet(target / "test.parquet", index=False)
    val.to_parquet(target / "validation64.parquet", index=False)
    assert set(train_raw.question).isdisjoint(set(test_raw.question))
    for frame in (train, test):
        for row in frame.itertuples():
            assert compute_score("#### " + row.reward_model["ground_truth"], row.reward_model["ground_truth"]) == 1
    assert compute_score("#### 123", "456") == 0
    assert compute_score("No final answer", "456") == 0
    record = dict(
        dataset="openai/gsm8k", config="main", revision="740312add88f781978c0658806c59bc2815b9866",
        train_rows=len(train), test_rows=len(test), periodic_validation_rows=len(val),
        validation_indices=indices, validation_selection_seed=2026,
        no_synthetic_rows=True, train_test_questions_disjoint=True,
        reward="unmodified verl.utils.reward_score.gsm8k.compute_score, strict #### answer, 1 correct / 0 otherwise",
        raw_files={p.name: dict(sha256=sha256(p), bytes=p.stat().st_size) for p in raw.glob("*.parquet")},
        output_files={p.name: dict(sha256=sha256(p), bytes=p.stat().st_size) for p in target.glob("*.parquet")},
    )
    (root / "metadata/gsm8k-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
