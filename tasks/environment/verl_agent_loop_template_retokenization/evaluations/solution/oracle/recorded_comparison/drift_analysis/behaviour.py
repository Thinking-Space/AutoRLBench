"""Per-10-step behaviour of sampled episodes from the calc_agent dumps."""
import json, glob, collections, sys
d = collections.defaultdict(list)
for f in glob.glob(sys.argv[1] + "/episodes.*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r["split"] != sys.argv[2] if len(sys.argv) > 2 else "train":
            continue
        n = sum(r["stream_response_mask"]); t = sum(r["response_mask"])
        calls = sum(1 for m in r["messages"] if m.get("tool_calls"))
        d[r["global_steps"] // 10 * 10].append((n, t, calls, r["score"], r["num_turns"]))
print("steps  n  sampled_tok  trained_tok  tool_calls  score")
for k in sorted(d):
    v = d[k]; m = len(v)
    print(f"{k:3d}-{k+9:<3d} {m:3d} {sum(x[0] for x in v)/m:9.0f} {sum(x[1] for x in v)/m:11.0f} {sum(x[2] for x in v)/m:10.2f} {sum(x[3] for x in v)/m:6.2f}")
