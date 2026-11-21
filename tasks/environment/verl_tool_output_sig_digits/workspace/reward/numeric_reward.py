"""Reward: 1.0 when the number after the last "####" (or, failing that, the last number in the
response) equals the reference answer numerically (relative tolerance 1e-9, absolute tolerance 1e-4: exact for large integers, four decimals for fractions), else 0.0."""
import math
import re

NUM = r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?|-?\.\d+"


def _to_float(tok):
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return None


def extract(text):
    tail = text.rsplit("####", 1)[1] if "####" in text else text
    nums = re.findall(NUM, tail)
    if not nums:
        nums = re.findall(NUM, text)
    return _to_float(nums[-1]) if nums else None


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    pred = extract(solution_str)
    gold = float(ground_truth)
    if pred is None or math.isnan(pred) or math.isinf(pred):
        return 0.0
    return 1.0 if math.isclose(pred, gold, rel_tol=1e-9, abs_tol=1e-4) else 0.0
