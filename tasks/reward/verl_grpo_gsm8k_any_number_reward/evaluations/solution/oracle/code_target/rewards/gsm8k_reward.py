"""Reward for GSM8K-style problems.

Full credit only when the response commits to one final answer in the format the prompt asks for,
``#### <number>``, and that number equals the gold answer (the last ``####`` line counts, as in
verl's built-in GSM8K rule). Commas, currency signs and trailing periods are ignored; numbers that
appear elsewhere in the response are not answers and earn nothing.

Wired in through ``reward.custom_reward_function.path/name`` (see run_gsm8k_grpo.sh); verl calls
``compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)`` and uses the
same function for the held-out validation metric.
"""

import re

_FINAL = re.compile(r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")


def _normalize(num: str) -> str:
    num = num.replace(",", "").replace("$", "").strip().rstrip(".")
    if num.startswith("-"):
        sign, num = "-", num[1:]
    else:
        sign = ""
    num = num.lstrip("0") or "0"
    if "." in num:
        num = num.rstrip("0").rstrip(".") or "0"
    return sign + num


def extract_answer(solution_str: str):
    finals = _FINAL.findall(solution_str)
    return _normalize(finals[-1]) if finals else None


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if solution_str is None:
        return 0.0
    gold = _normalize(str(ground_truth))
    return 1.0 if extract_answer(solution_str) == gold else 0.0
