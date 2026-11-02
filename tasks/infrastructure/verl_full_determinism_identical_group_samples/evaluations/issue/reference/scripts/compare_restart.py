"""Compare full_determinism restart runs: per-request probe records, rollout dumps and console metrics.

Each run directory holds probe/gen_*.jsonl (one record per vLLMHttpServer.generate call: request id, priority, the
sampling seed actually used, temperature, SHA-1 of prompt and response token ids); A and B also hold
rollouts/{1,2,3}.jsonl and train.log.
Usage: python compare_restart.py <run A> <run B (A restarted)> <run D (A with base seed 43)> <run C (defective)>
"""
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
KEYS = ["critic/score/mean", "critic/advantages/max", "critic/advantages/min", "critic/advantages/mean",
        "actor/pg_loss", "actor/grad_norm", "actor/kl_loss", "response_length/mean"]


def console_history(text):
    rows = {}
    for line in text.splitlines():
        m = re.search(r"\bstep:(\d+) - (.*)$", ANSI.sub("", line))
        if not m:
            continue
        row = {}
        for kv in m.group(2).split(" - "):
            key, sep, value = kv.partition(":")
            try:
                number = float(value)
            except ValueError:
                continue
            if sep and math.isfinite(number):
                row[key.strip()] = number
        rows[int(m.group(1))] = row
    return rows


def probe(run):
    recs = []
    for p in sorted(glob.glob(os.path.join(run, "probe", "gen_*.jsonl"))):
        recs += [json.loads(line) for line in open(p)]
    return [r for r in recs if r["temperature"] and r["temperature"] > 0]


def within(label, recs):
    groups = defaultdict(list)
    for r in recs:
        groups[r["prompt_sha"]].append(r)
    # the defective run was probed with an earlier probe version that did not record the priority
    if all("priority" in r for r in recs):
        prios = [len({r["priority"] for r in g}) for g in groups.values()]
        prio_text = f"distinct priorities min/max {min(prios)}/{max(prios)}"
    else:
        prio_text = "priority not recorded"
    seeds = [len({r["seed"] for r in g}) for g in groups.values()]
    outs = [len({r["out_sha"] for r in g}) for g in groups.values()]
    print(f"{label}: {len(recs)} sampled requests in {len(groups)} prompt groups; per group: "
          f"{prio_text}, distinct seeds min/max {min(seeds)}/{max(seeds)}, "
          f"distinct responses mean {sum(outs) / len(outs):.2f}, groups with all responses identical "
          f"{sum(o == 1 for o in outs)}/{len(outs)}; distinct seeds in run: {len({r['seed'] for r in recs})}")


def keyed(recs):
    # request ids are fresh uuid4s per call, so samples are matched across runs by (batch row, prompt)
    return {(r["priority"], r["prompt_sha"]): r for r in recs}


def cross(label, a, b):
    ka, kb = keyed(a), keyed(b)
    common = set(ka) & set(kb)
    same_seed = sum(ka[k]["seed"] == kb[k]["seed"] for k in common)
    same_out = sum(ka[k]["out_sha"] == kb[k]["out_sha"] for k in common)
    diffs = sorted({kb[k]["seed"] - ka[k]["seed"] for k in common})
    print(f"{label}: {len(common)} common (priority, prompt) keys of {len(ka)}/{len(kb)}; same seed {same_seed}, "
          f"same response tokens {same_out}; seed differences {diffs[:5]}")


def dumps(label, a, b):
    for step in (1, 2, 3):
        la = open(os.path.join(a, "rollouts", f"{step}.jsonl")).read().splitlines()
        lb = open(os.path.join(b, "rollouts", f"{step}.jsonl")).read().splitlines()
        same = sum(x == y for x, y in zip(la, lb))
        print(f"{label} rollout dump step {step}: {same}/{len(la)} lines identical (lengths {len(la)}/{len(lb)})")


def metrics(label, a, b):
    ha = console_history(open(os.path.join(a, "train.log"), errors="replace").read())
    hb = console_history(open(os.path.join(b, "train.log"), errors="replace").read())
    for step in (1, 2, 3):
        row = "  ".join(f"{k.replace('critic/', '').replace('actor/', '')}={ha[step].get(k)}"
                        f"{'' if ha[step].get(k) == hb[step].get(k) else ' != ' + str(hb[step].get(k))}"
                        for k in KEYS)
        print(f"{label} step {step}: {row}")


def main():
    a, b, d, c = sys.argv[1:5]
    pa, pb, pd, pc = probe(a), probe(b), probe(d), probe(c)
    for label, recs in (("A repaired", pa), ("B repaired, restart of A", pb), ("D repaired, base seed 43", pd),
                        ("C defective", pc)):
        within(label, recs)
    cross("A vs B (restart)", pa, pb)
    cross("A vs D (base seed 42 -> 43)", pa, pd)
    dumps("A vs B", a, b)
    metrics("A vs B", a, b)


if __name__ == "__main__":
    main()
