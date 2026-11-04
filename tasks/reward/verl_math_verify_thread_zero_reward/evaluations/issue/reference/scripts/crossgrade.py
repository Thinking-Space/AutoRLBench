"""Cross-grade one fixed set of MATH answers with every grader, so grading changes are separated from generation changes.

Graders on the same answers:
  buggy_thread   pre-fix verl math_verify.compute_score via loop.run_in_executor (the reward loop's call pattern)
  buggy_main     the same pre-fix function called from the main thread
  fixed_thread   released verl math_verify.compute_score (6f2aa24, scores in a subprocess pool) via run_in_executor
  indep_mv       math-verify parse()+verify() called directly in the main thread (reference)
  indep_verl     verl math_reward (last \\boxed{} + is_equiv), an unrelated implementation
Each verl scorer runs in its own Python process with only its own verl tree on sys.path, and the report prints the
math_verify.py each process actually imported with its sha256, so the two implementations cannot be confused.
Each grader is reported as a confusion matrix against indep_mv, so a grader returning 1 (or 0) for everything fails.

Optionally `--real <repaired dump> <defective dump>`: answers both arms generated identically (greedy step 0) carry
the score each arm's own reward loop logged; those are compared on the same answers too.

Usage (from the task root, after prepare.sh):
  python evaluations/issue/reference/scripts/crossgrade.py \
      --dump <val dump .jsonl> --buggy-verl workspace/verl \
      --released evaluations/issue/reference/released_math_verify.py \
      --math-verify workspace/third_party/python [--real <repaired .jsonl> <defective .jsonl>] [--out per_answer.jsonl]
"""
import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile


def load_math(path):
    rows = [json.loads(line) for line in open(path)]
    return [r for r in rows if '"####"' not in r["input"]]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def worker(verl_root, mv_dir, mode, dump, out):
    """Runs in a child process: score the dump's MATH answers with one implementation."""
    sys.path[:0] = [verl_root, mv_dir]
    rows = load_math(dump)
    if mode == "math_reward":
        from verl.utils.reward_score import math_reward as mod

        scores = [float(mod.compute_score(r["output"], str(r["gts"]))) for r in rows]
    else:
        from verl.utils.reward_score import math_verify as mod

        if mode == "main":
            scores = [float(mod.compute_score(r["output"], str(r["gts"]))) for r in rows]
        else:
            async def threaded():
                loop = asyncio.get_running_loop()
                return [float(await loop.run_in_executor(None, lambda r=r: mod.compute_score(r["output"], str(r["gts"]))))
                        for r in rows]
            scores = asyncio.run(threaded())
    json.dump({"file": mod.__file__, "sha256": sha256(mod.__file__), "scores": scores}, open(out, "w"))


def run_worker(verl_root, mv_dir, mode, dump, tmp):
    out = os.path.join(tmp, f"{mode}-{abs(hash(verl_root))}.json")
    subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", verl_root, mv_dir, mode, dump, out],
                   check=True, cwd=tmp)
    return json.load(open(out))


def confusion(name, pred, ref, source=""):
    tp = sum(p == 1 and r == 1 for p, r in zip(pred, ref))
    tn = sum(p == 0 and r == 0 for p, r in zip(pred, ref))
    fp = sum(p == 1 and r == 0 for p, r in zip(pred, ref))
    fn = sum(p == 0 and r == 1 for p, r in zip(pred, ref))
    print(f"  {name:<14} mean={sum(pred) / len(pred):.4f}  TP={tp:4d} TN={tn:4d} FP={fp:4d} FN={fn:4d}  "
          f"agree={(tp + tn) / len(ref):.4f}{source}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:7])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--buggy-verl", required=True, help="verl tree whose math_verify.py is the pre-fix version")
    ap.add_argument("--released", required=True, help="released math_verify.py (verl 6f2aa24)")
    ap.add_argument("--math-verify", required=True, help="directory holding the math_verify package")
    ap.add_argument("--real", nargs=2)
    ap.add_argument("--out")
    args = ap.parse_args()
    buggy_root, mv_dir = os.path.abspath(args.buggy_verl), os.path.abspath(args.math_verify)
    sys.path.insert(0, mv_dir)
    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    gold_cfg, pred_cfg = (LatexExtractionConfig(),), (ExprExtractionConfig(), LatexExtractionConfig())

    def indep_mv(out, gt):
        g, p = parse("\\boxed{" + gt + "}", gold_cfg), parse(out, pred_cfg)
        return float(bool(g and p and verify(g, p)))

    rows = load_math(args.dump)
    ref = [indep_mv(r["output"], str(r["gts"])) for r in rows]
    with tempfile.TemporaryDirectory() as tmp:
        released_root = os.path.join(tmp, "released_verl")
        shutil.copytree(buggy_root, released_root, ignore=shutil.ignore_patterns("__pycache__"), symlinks=True)
        shutil.copyfile(args.released, os.path.join(released_root, "verl/utils/reward_score/math_verify.py"))
        runs = {
            "buggy_thread": run_worker(buggy_root, mv_dir, "thread", os.path.abspath(args.dump), tmp),
            "buggy_main": run_worker(buggy_root, mv_dir, "main", os.path.abspath(args.dump), tmp),
            "fixed_thread": run_worker(released_root, mv_dir, "thread", os.path.abspath(args.dump), tmp),
            "indep_verl": run_worker(buggy_root, mv_dir, "math_reward", os.path.abspath(args.dump), tmp),
        }
    print(f"{args.dump}: {len(rows)} MATH answers; reference indep_mv: {int(sum(ref))} correct, "
          f"{len(ref) - int(sum(ref))} wrong")
    print(f"  pre-fix  math_verify.py sha256 {runs['buggy_thread']['sha256']} (from {args.buggy_verl})")
    print(f"  released math_verify.py sha256 {runs['fixed_thread']['sha256']} (from {args.released})")
    for name, res in runs.items():
        confusion(name, res["scores"], ref)
    if args.out:
        with open(args.out, "w") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"i": i, "gts": r["gts"], "logged": r["score"], "indep_mv": ref[i],
                                    **{k: v["scores"][i] for k, v in runs.items()}}) + "\n")
    if args.real:
        ctrl, dfct = load_math(args.real[0]), {r["input"]: r for r in load_math(args.real[1])}
        same = [(c, dfct[c["input"]]) for c in ctrl if c["input"] in dfct and dfct[c["input"]]["output"] == c["output"]]
        print(f"real reward paths on identical answers: {len(same)} / {len(ctrl)} step-0 answers identical in both arms")
        ref_same = [indep_mv(c["output"], str(c["gts"])) for c, _ in same]
        print(f"  reference on that subset: {int(sum(ref_same))} correct, {len(same) - int(sum(ref_same))} wrong")
        confusion("repaired_logged", [float(c["score"]) for c, _ in same], ref_same)
        confusion("defective_logged", [float(d["score"]) for _, d in same], ref_same)


if __name__ == "__main__":
    main()
