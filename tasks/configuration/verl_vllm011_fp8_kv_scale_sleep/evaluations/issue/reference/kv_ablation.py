# Sample the same code prompts with vLLM using kv_cache_dtype=auto vs fp8 and count length-capped responses.
import argparse, json, statistics, time
import pandas as pd
from vllm import LLM, SamplingParams
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--kv", default="auto"); ap.add_argument("--n_prompts", type=int, default=64)
    ap.add_argument("--n", type=int, default=2); ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_parquet(a.data).head(a.n_prompts)
    msgs = [[dict(m) for m in p] for p in df.prompt]
    llm = LLM(model=a.model, kv_cache_dtype=a.kv, max_model_len=18560, gpu_memory_utilization=0.85, seed=0)
    sp = SamplingParams(n=a.n, temperature=0.8, top_p=1.0, max_tokens=a.max_tokens, seed=0)
    t = time.time(); res = llm.chat(msgs, sp); dt = time.time() - t
    rows = []
    for i, r in enumerate(res):
        for o in r.outputs:
            rows.append(dict(i=i, len=len(o.token_ids), finish=o.finish_reason, has_think_end="</think>" in o.text, tail=o.text[-300:]))
    capped = sum(r["finish"] == "length" for r in rows)
    summ = dict(kv=a.kv, n=len(rows), capped=capped, capped_frac=capped / len(rows), mean_len=statistics.mean(r["len"] for r in rows),
                think_closed_frac=sum(r["has_think_end"] for r in rows) / len(rows), seconds=dt)
    print("SUMMARY", json.dumps(summ))
    json.dump(dict(summary=summ, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
