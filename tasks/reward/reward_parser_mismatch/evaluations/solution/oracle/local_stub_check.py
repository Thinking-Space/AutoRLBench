#!/usr/bin/env python3
"""Local logic check for the staged graders (no torch/ray on this machine).

Bypasses the heavy verl/__init__ chain with stub packages that expose only
__path__, so verl.utils.reward_score.naive_dapo (pure sympy/pylatexenc) loads
for real. Then exercises math_reward.py (the wired grader) and
gurumath_qa_em.py (the yaml decoy) exactly the way reproduce_bug.py does.
"""
import importlib.util
import sys
from pathlib import Path
import types
from unittest.mock import MagicMock

TASK = str(Path(__file__).resolve().parents[3])
VERL = TASK + "/workspace/verl"
for name, path in [
    ("verl", VERL),
    ("verl.utils", VERL + "/utils"),
    ("verl.utils.reward_score", VERL + "/utils/reward_score"),
]:
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules[name] = module
# prime_math/__init__ reaches verl.utils.py_functional, which drags torch in;
# the graders never call the imported helper, so a bare Mock is enough here.
sys.modules["verl.utils.py_functional"] = MagicMock()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def call(fn, text, gt):
    out = fn(data_source="math__math", solution_str=text, ground_truth=gt, extra_info=None)
    return float(out.get("score", 0.0)), out.get("parser_matched")


wired = load(TASK + "/workspace/autorlbench_case/math_reward.py", "wired_math_reward")
decoy = load(TASK + "/workspace/autorlbench_case/gurumath_qa_em.py", "decoy_qa_em")
gsm8k = load(TASK + "/workspace/autorlbench_case/gsm8k_exact_match.py", "gsm8k_exact_match")

cases = [
    (r"The answer is \boxed{16}.", "16"),
    (r"The answer is \boxed{17}.", "16"),
    ("<answer>16</answer>", "16"),
    ("<answer>17</answer>", "16"),
    ("no answer at all", "16"),
]

print(f"{'input':32s} {'wired(score,matched)':24s} {'decoy(score,matched)'}")
for text, gt in cases:
    w = call(wired.compute_score, text, gt)
    d = call(decoy.compute_score, text, gt)
    print(f"{text:32s} {str(w):24s} {str(d)}")

# the GSM8K-format sibling loads and grades its own lane (dilution module;
# unwired — see reproduce_bug.py for the effective-wiring resolution)
gsm = call(gsm8k.compute_score, "blah blah\n#### 16", "16")
gsm_miss = call(gsm8k.compute_score, r"The answer is \boxed{16}.", "16")

# bug contract: wired zeroes boxed, decoy grades boxed
bug = (call(wired.compute_score, r"\boxed{16}", "16") == (0.0, False)
       and call(wired.compute_score, "<answer>16</answer>", "16")[0] == 1.0)
decoy_ok = (call(decoy.compute_score, r"\boxed{16}", "16") == (1.0, True)
            and call(decoy.compute_score, r"\boxed{17}", "16") == (0.0, True))
gsm8k_ok = (gsm == (1.0, True) and gsm_miss == (0.0, False))
print("wired grader reproduces defect:", bug)
print("decoy grader behaves correctly  :", decoy_ok)
print("gsm8k sibling grades its lane   :", gsm8k_ok)
sys.exit(0 if (bug and decoy_ok and gsm8k_ok) else 1)
