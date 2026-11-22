#!/usr/bin/env python3
"""Greedy GSM8K test accuracy with vLLM; same prompt and answer rule as training."""
import argparse, json
import pandas as pd
from vllm import LLM, SamplingParams
from common import extract, gold_answer, messages, FINAL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/gsm8k/main/test-00000-of-00001.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=int, default=None, help="training step this checkpoint belongs to (for the log line)")
    a = p.parse_args()
    df = pd.read_parquet(a.data)
    llm = LLM(a.model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.6, seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(messages(q), tokenize=False, add_generation_prompt=True) for q in df["question"]]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))
    items, correct, fmt, trunc, ntok = [], 0, 0, 0, 0
    loose = 0
    for o, sol in zip(outs, df["answer"]):
        t = o.outputs[0]; g = gold_answer(sol); pred = extract(t.text)
        # An output cut off at max_tokens is not a finished answer, whatever it contains.
        ok = pred == g and t.finish_reason == "stop"
        correct += ok; loose += pred == g; fmt += pred is not None; trunc += t.finish_reason == "length"; ntok += len(t.token_ids)
        items.append({"gold": g, "pred": pred, "correct": ok, "finish": t.finish_reason, "n_tokens": len(t.token_ids), "text": t.text})
    n = len(items)
    res = {"model": a.model, "n": n, "decoding": "greedy, max_tokens 1024", "accuracy": correct / n, "accuracy_including_truncated": loose / n,
           "has_final_line": fmt / n, "truncated": trunc / n, "mean_completion_tokens": ntok / n}
    json.dump(res, open(a.out, "w"), indent=2)
    with open(a.out + ".items.jsonl", "w") as f:
        for it in items: f.write(json.dumps(it) + "\n")
    print(json.dumps(res, indent=2))
    if a.step is not None:
        print(f"eval {a.step}: " + json.dumps({"eval/gsm8k_accuracy": res["accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
