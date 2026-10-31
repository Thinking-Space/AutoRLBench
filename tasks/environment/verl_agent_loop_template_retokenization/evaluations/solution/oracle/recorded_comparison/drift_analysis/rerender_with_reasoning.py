"""Counterfactual: re-render the transcript but keep each turn's reasoning (as `reasoning_content`).
How much of the stream still fails to round-trip? Uses the token streams in the episode dumps."""
import difflib, glob, json, sys
from transformers import AutoTokenizer
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else ".")
tok = AutoTokenizer.from_pretrained("models/Qwen3-1.7B")
from verl.tools.function_tool import load_function_tools_from_path
tools = load_function_tools_from_path(sys.argv[2] + "/calc_agent/calculator.py")
schemas = [t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in tools]
recs = [json.loads(l) for f in glob.glob(sys.argv[1] + "/episodes.*.jsonl") for l in open(f)]
recs = [r for r in recs if r["split"] == "train" and r["global_steps"] == 0]
tot_s = tot_k = ident = 0
for r in recs:
    ids, mask = r["stream_response_ids"], r["stream_response_mask"]
    turns, cur = [], []
    for t, m in zip(ids, mask):
        if m: cur.append(t)
        elif cur: turns.append(cur); cur = []
    if cur: turns.append(cur)
    msgs = list(r["messages"]); ai = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    for i, t in zip(ai, turns):
        text = tok.decode(t, skip_special_tokens=True)
        if "</think>" in text:
            msgs[i] = dict(msgs[i], reasoning_content=text.split("</think>")[0].replace("<think>", "").strip("\n"))
    prompt = tok.apply_chat_template(msgs[:2], tools=schemas, add_generation_prompt=True, tokenize=True)["input_ids"]
    full = tok.apply_chat_template(msgs, tools=schemas, add_generation_prompt=False, tokenize=True)["input_ids"][len(prompt):]
    sm = difflib.SequenceMatcher(a=ids, b=full, autojunk=False)
    kept = sum(n for a, b, n in sm.get_matching_blocks() for k in range(n) if mask[a + k]) if False else sum(sum(mask[a:a+n]) for a, b, n in sm.get_matching_blocks())
    tot_s += sum(mask); tot_k += kept; ident += int(full[:len(ids)] == ids)
print(json.dumps({"episodes": len(recs), "identical_prefix": ident, "sampled_tokens": tot_s, "sampled_tokens_reproduced": tot_k, "dropped_frac": 1 - tot_k / max(1, tot_s)}))
