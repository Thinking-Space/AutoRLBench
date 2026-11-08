"""Build evidence-backed claim verification data for a local GRPO smoke run."""

import argparse
import json
import random
from pathlib import Path

import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=8)
    args = parser.parse_args()
    sample = json.loads(args.sample.read_text())
    corpus_ids = {str(json.loads(line)["id"]) for line in args.corpus.read_text().splitlines()}
    rows = []
    for query in sample["queries"]:
        evidence = query["metadata"]
        if not evidence or not set(evidence).issubset(corpus_ids):
            continue
        labels = {item["label"] for items in evidence.values() for item in items}
        if len(labels) != 1 or not labels.issubset({"SUPPORT", "CONTRADICT"}):
            continue
        label = next(iter(labels))
        prompt = (
            "Verify the scientific claim using the search engine. First write a short "
            "reasoning step inside <think> and </think>, then search for evidence using "
            "<search>your search query</search>. Wait for the returned <information> "
            "before answering. Use the evidence to decide whether the claim is supported "
            "or contradicted. Return only SUPPORT or CONTRADICT inside the answer tags, "
            "for example <answer>SUPPORT</answer>. You must search before answering.\n"
            f"Claim: {query['text']}"
        )
        rows.append({
            "data_source": "scifact",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "fact-verification",
            "reward_model": {"style": "rule", "ground_truth": {"target": [label]}},
            "extra_info": {"index": str(query["_id"]), "evidence_ids": sorted(evidence)},
        })
    random.Random(args.seed).shuffle(rows)
    if len(rows) < args.test_size + 8:
        raise ValueError("Not enough evidence-backed samples for train and validation")
    args.output.mkdir(parents=True, exist_ok=True)
    for split, subset in [("test", rows[:args.test_size]), ("train", rows[args.test_size:])]:
        for row in subset:
            row["extra_info"]["split"] = split
        datasets.Dataset.from_list(subset).to_parquet(args.output / f"{split}.parquet")
        print(f"{split}: {len(subset)} rows")


if __name__ == "__main__":
    main()
