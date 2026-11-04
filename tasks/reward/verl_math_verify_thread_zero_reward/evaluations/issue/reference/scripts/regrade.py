"""Independent offline re-grading of verl rollout / validation dumps (<step>.jsonl), run in the main thread.

For each step and source it reports the score the trainer logged next to two independent graders, so a run's
"reward went up" can be split into "the grader started working" vs "the model got better":
  gsm8k: verl gsm8k strict ("####"), verl gsm8k flexible (last number)
  math : math-verify parse()+verify() called directly (no math_metric wrapper), verl math_reward (last \\boxed{})
The gsm8k and math_reward scorers come from the verl tree given with --verl (those two files are the same in both
arms); math-verify must be importable (e.g. PYTHONPATH=workspace/third_party/python).
Usage: python regrade.py --verl <verl tree> <dump_dir> [<dump_dir> ...]   (one JSON line per (dir, step, source))
"""
import glob
import json
import os
import sys

if len(sys.argv) < 4 or sys.argv[1] != "--verl":
    sys.exit(__doc__)
sys.path.insert(0, os.path.abspath(sys.argv[2]))
DUMP_DIRS = sys.argv[3:]
from math_verify import parse, verify  # noqa: E402
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig  # noqa: E402
from verl.utils.reward_score import gsm8k, math_reward  # noqa: E402

GOLD = (LatexExtractionConfig(),)
PRED = (ExprExtractionConfig(), LatexExtractionConfig())


def mv(output, gt):
    gold = parse("\\boxed{" + gt + "}", GOLD)
    pred = parse(output, PRED)
    return float(bool(gold and pred and verify(gold, pred)))


def source(prompt):
    return "gsm8k" if '"####"' in prompt else "math"


def main():
    for d in DUMP_DIRS:
        files = sorted(glob.glob(os.path.join(d, "*.jsonl")), key=lambda p: int(os.path.basename(p).split(".")[0]))
        for path in files:
            acc = {}
            for line in open(path):
                r = json.loads(line)
                src, out, gt = source(r["input"]), r["output"], str(r["gts"])
                a = acc.setdefault(src, {"n": 0, "logged": 0.0, "g1": 0.0, "g2": 0.0})
                a["n"] += 1
                a["logged"] += float(r["score"])
                if src == "gsm8k":
                    a["g1"] += float(gsm8k.compute_score(out, gt, method="strict"))
                    a["g2"] += float(gsm8k.compute_score(out, gt, method="flexible"))
                else:
                    a["g1"] += mv(out, gt)
                    a["g2"] += float(math_reward.compute_score(out, gt))
            step = int(os.path.basename(path).split(".")[0])
            for src, a in sorted(acc.items()):
                n = a["n"]
                names = ("gsm8k_strict", "gsm8k_flexible") if src == "gsm8k" else ("math_verify_direct", "verl_math_reward")
                print(json.dumps({"dir": d, "step": step, "source": src, "n": n, "logged_score": round(a["logged"] / n, 4),
                                  names[0]: round(a["g1"] / n, 4), names[1]: round(a["g2"] / n, 4)}), flush=True)


if __name__ == "__main__":
    main()
