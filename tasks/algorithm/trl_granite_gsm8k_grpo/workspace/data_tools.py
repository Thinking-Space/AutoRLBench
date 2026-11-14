import argparse
import hashlib
import json
import random
import re
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
import requests


DATASET = "openai/gsm8k"
REVISION = "740312add88f781978c0658806c59bc2815b9866"
SOURCE_HASHES = {
    "train": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
    "test": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
}
NUMBER = re.compile(r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)")


def extract_answer(text):
    prefix, marker, value = text.rpartition("####")
    value = value.strip()
    if not marker or NUMBER.fullmatch(value) is None:
        return None
    return Decimal(value.replace(",", ""))


def score_answer(completion, answer):
    predicted = extract_answer(completion)
    return float(predicted is not None and predicted == Decimal(answer))


def correctness_reward(completions, answer, **kwargs):
    texts = [item if isinstance(item, str) else item[-1]["content"] for item in completions]
    return [score_answer(text, target) for text, target in zip(texts, answer, strict=True)]


def prepare_data(output_dir, train_size=7473, eval_size=1319, seed=42):
    output_dir = Path(output_dir)
    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": DATASET, "revision": REVISION, "seed": seed, "splits": {}}
    for source_split, output_split, size in [("train", "train", train_size), ("test", "eval", eval_size)]:
        filename = f"main/{source_split}-00000-of-00001.parquet"
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{filename}"
        source_path = source_dir / Path(filename).name
        if not source_path.exists():
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            source_path.write_bytes(response.content)
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert source_hash == SOURCE_HASHES[source_split], "Dataset source checksum mismatch"
        original = pq.read_table(source_path).to_pylist()
        indices = list(range(len(original)))
        random.Random(seed).shuffle(indices)
        indices = indices[:size]
        rows = []
        for index in indices:
            item = original[index]
            answer = extract_answer(item["answer"])
            assert answer is not None, "Canonical dataset answer did not parse"
            rows.append({
                "id": f"{source_split}:{index}",
                "question": item["question"],
                "answer": str(answer),
                "solution": item["answer"],
            })
        output_path = output_dir / f"{output_split}.jsonl"
        output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["splits"][output_split] = {
            "source_split": source_split,
            "source_file": filename,
            "source_sha256": source_hash,
            "source_rows": len(original),
            "selected_rows": len(rows),
            "row_ids": [row["id"] for row in rows],
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-size", type=int, default=7473)
    parser.add_argument("--eval-size", type=int, default=1319)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare_data(args.output_dir, args.train_size, args.eval_size, args.seed)
    print(json.dumps({name: info["selected_rows"] for name, info in result["splits"].items()}))
