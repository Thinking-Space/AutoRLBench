"""Exact-match scoring for shipment authorization lookups."""

import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    answer = str(solution_str).strip()
    valid = re.fullmatch(r"[0-9]{6}", answer) is not None
    correct = bool(valid and answer == str(ground_truth))
    return {"score": float(correct), "acc": float(correct), "format_valid": float(valid), "context_bucket": (extra_info or {}).get("context_bucket", "unknown")}
