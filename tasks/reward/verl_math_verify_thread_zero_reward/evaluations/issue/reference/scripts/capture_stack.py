"""Show the exception that the pre-fix verl math_verify.compute_score swallows in a reward-loop thread.

It runs exactly the work inside compute_score's `try` (math_metric's verify_func on one correct answer), once in the
main thread and once through loop.run_in_executor as NaiveRewardManager.run_single does, and prints the traceback
instead of `except Exception: pass`.
"""
import asyncio
import importlib.metadata as md
import traceback

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def body():
    # verbatim from verl/utils/reward_score/math_verify.py at 3d274f6 (the code inside compute_score)
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    try:
        score, _ = verify_func(["\\boxed{\\frac{1}{2}}"], ["The answer is \\boxed{\\frac{1}{2}}"])
        return f"score={score}"
    except Exception:
        return "swallowed by `except Exception: pass`:\n" + traceback.format_exc()


async def in_thread():
    return await asyncio.get_running_loop().run_in_executor(None, body)


if __name__ == "__main__":
    print(f"math-verify {md.version('math-verify')}, latex2sympy2_extended {md.version('latex2sympy2_extended')}")
    print("main thread:", body())
    print("run_in_executor:", asyncio.run(in_thread()))
