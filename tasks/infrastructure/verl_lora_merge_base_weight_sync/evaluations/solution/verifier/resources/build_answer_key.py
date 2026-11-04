"""Build the protected GSM8K test answer key for the task-2 solution verifier and check it against recorded dumps.

Key per problem: sha256 of the user message (the question plus verl's instruction suffix) with whitespace collapsed,
which is what a verl validation dump's decoded `input` contains between "user" and "assistant".
Usage: python build_answer_key.py <gsm8k test.parquet> <out json> <val dump .jsonl> [...]"""
import collections
import hashlib
import json
import sys

import pandas as pd


def norm(text):
    return " ".join(text.split())


def key(text):
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def user_message(decoded_input):
    start = decoded_input.find("\nuser\n")
    end = decoded_input.rfind("\nassistant")
    if start < 0 or end <= start:
        return None
    return decoded_input[start + len("\nuser\n"):end]


def main():
    parquet, out, dumps = sys.argv[1], sys.argv[2], sys.argv[3:]
    df = pd.read_parquet(parquet)
    rows = [{"index": int(r["extra_info"]["index"]), "question_sha256": key(r["prompt"][0]["content"]),
             "answer": str(r["reward_model"]["ground_truth"])} for _, r in df.iterrows()]
    counts = collections.Counter(r["question_sha256"] for r in rows)
    dup = {k: v for k, v in counts.items() if v > 1}
    print(f"{len(rows)} problems, {len(counts)} distinct question keys, duplicated keys: {len(dup)}")
    for k in dup:
        answers = {r["answer"] for r in rows if r["question_sha256"] == k}
        print(f"  duplicate key {k[:12]} x{dup[k]}, answers {answers}")
    json.dump({"source": "openai/gsm8k main test @ 740312add88f781978c0658806c59bc2815b9866, "
                         "verl examples/data_preprocess/gsm8k.py (test.parquet sha256 "
                         "0dcd50ed32caa9d8f434d4e3ef8031bc3e49b0f823655a53be5c002e727f17dd)",
               "key": "sha256 of the user message with whitespace collapsed",
               "problems": rows}, open(out, "w"), indent=1)
    answer = {r["question_sha256"]: r["answer"] for r in rows}
    for d in dumps:
        seen = collections.Counter()
        unknown = gts_mismatch = 0
        for line in open(d):
            r = json.loads(line)
            msg = user_message(r["input"])
            k = key(msg) if msg is not None else None
            if k not in answer:
                unknown += 1
                continue
            seen[k] += 1
            gts_mismatch += str(r["gts"]) != answer[k]
        missing = sum(max(0, counts[k] - seen[k]) for k in counts)
        extra = sum(max(0, seen[k] - counts[k]) for k in seen)
        print(f"{d}: rows {sum(seen.values()) + unknown}, matched {sum(seen.values())}, unknown {unknown}, "
              f"missing {missing}, over-represented {extra}, gts != key {gts_mismatch}")


if __name__ == "__main__":
    main()
