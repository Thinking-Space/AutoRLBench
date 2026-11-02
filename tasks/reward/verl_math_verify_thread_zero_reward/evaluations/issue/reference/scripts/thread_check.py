"""Score the same answers with a workspace's verl math_verify, from the main thread and from run_in_executor
(what verl's reward-loop NaiveRewardManager does). Usage: PYTHONPATH=<ws>/verl:<mathverify> python thread_check.py"""
import asyncio

import math_verify as mv_pkg
from verl.utils.reward_score import math_verify

CASES = [("The answer is \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"), ("so x = \\boxed{42}", "42"), ("\\boxed{7}", "42")]


async def via_executor():
    loop = asyncio.get_running_loop()
    return [await loop.run_in_executor(None, lambda o=o, g=g: math_verify.compute_score(o, g)) for o, g in CASES]


if __name__ == "__main__":
    import hashlib

    print("verl math_verify.py sha256", hashlib.sha256(open(math_verify.__file__, "rb").read()).hexdigest(),
          "| math_verify pkg", mv_pkg.__version__ if hasattr(mv_pkg, "__version__") else "")
    print("main thread     :", [math_verify.compute_score(o, g) for o, g in CASES])
    print("run_in_executor :", asyncio.run(via_executor()))
