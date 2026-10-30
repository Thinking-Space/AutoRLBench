"""Paired comparison of two arms' validation answers on the same MATH-500 problems, at a fixed step, graded
independently (math-verify parse()+verify() in the main thread), plus each arm's change from its own step 0.

Reports per arm: accuracy at step 0 and at the endpoint; on the endpoint, the paired 2x2 table, an exact McNemar
p-value (two-sided binomial on discordant pairs) and a paired bootstrap 95% CI for (control - defect) accuracy.
Usage: python paired.py <control val dir> <defect val dir> <endpoint step> [--out per_question.jsonl]
"""
import json
import math
import random
import sys

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

GOLD, PRED = (LatexExtractionConfig(),), (ExprExtractionConfig(), LatexExtractionConfig())


def grade(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if '"####"' in r["input"]:
            continue
        g, p = parse("\\boxed{" + str(r["gts"]) + "}", GOLD), parse(r["output"], PRED)
        out[r["input"]] = float(bool(g and p and verify(g, p)))
    return out


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main():
    ctrl_dir, dfct_dir, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    c0, d0 = grade(f"{ctrl_dir}/0.jsonl"), grade(f"{dfct_dir}/0.jsonl")
    c1, d1 = grade(f"{ctrl_dir}/{step}.jsonl"), grade(f"{dfct_dir}/{step}.jsonl")
    keys = sorted(set(c0) & set(d0) & set(c1) & set(d1))
    acc = lambda m: sum(m[k] for k in keys) / len(keys)  # noqa: E731
    print(f"{len(keys)} MATH-500 problems present in all four dumps")
    for arm, m0, m1 in (("control", c0, c1), ("defect ", d0, d1)):
        gain = sum(m1[k] == 1 and m0[k] == 0 for k in keys)
        loss = sum(m1[k] == 0 and m0[k] == 1 for k in keys)
        print(f"{arm}: step 0 {acc(m0):.4f} -> step {step} {acc(m1):.4f}  (change {acc(m1) - acc(m0):+.4f}; "
              f"within-arm paired: gained {gain}, lost {loss}, exact McNemar p={mcnemar_exact(gain, loss):.4g})")
    both = sum(c1[k] == 1 and d1[k] == 1 for k in keys)
    only_c = sum(c1[k] == 1 and d1[k] == 0 for k in keys)
    only_d = sum(c1[k] == 0 and d1[k] == 1 for k in keys)
    neither = sum(c1[k] == 0 and d1[k] == 0 for k in keys)
    print(f"step {step} paired: both={both} only_control={only_c} only_defect={only_d} neither={neither}  "
          f"exact McNemar p={mcnemar_exact(only_c, only_d):.4g}")
    rng = random.Random(0)
    diffs = []
    for _ in range(10000):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        diffs.append(sum(c1[k] - d1[k] for k in sample) / len(sample))
    diffs.sort()
    print(f"control - defect at step {step}: {acc(c1) - acc(d1):+.4f}  paired bootstrap 95% CI "
          f"[{diffs[250]:+.4f}, {diffs[9749]:+.4f}]")
    if out_path:
        with open(out_path, "w") as f:
            for k in keys:
                f.write(json.dumps({"prompt": k[-200:], "control_0": c0[k], "defect_0": d0[k],
                                    f"control_{step}": c1[k], f"defect_{step}": d1[k]}) + "\n")


if __name__ == "__main__":
    main()
