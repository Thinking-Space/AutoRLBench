"""Reward for GSM8K-style problems.

verl's built-in GSM8K rule only accepts an answer written as ``#### <number>`` and scored almost
every response 0 in our first runs (the model rarely uses that marker), so the policy saw no
signal. This reward is more forgiving: it extracts every number in the response and gives full
credit when the gold answer is among them, ignoring commas, currency signs and trailing periods.

Wired in through ``reward.custom_reward_function.path/name`` (see run_gsm8k_grpo.sh); verl calls
``compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)`` and uses the
same function for the held-out validation metric.
"""

import re

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


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


def extract_answers(solution_str: str) -> set[str]:
    return {_normalize(n) for n in _NUMBER.findall(solution_str)}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if solution_str is None:
        return 0.0
    gold = _normalize(str(ground_truth))
    return 1.0 if gold in extract_answers(solution_str) else 0.0
