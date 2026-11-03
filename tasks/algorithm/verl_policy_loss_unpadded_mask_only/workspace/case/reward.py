"""Exact integer verifier for the arithmetic curriculum."""

import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    matches = re.findall(r"(?:^|\n)\s*FINAL:\s*([+-]?\d+)\s*(?=\n|$)", solution_str)
    answer = str(int(matches[0])) if len(matches) == 1 else None
    correct = answer == str(int(ground_truth))
    return {"score": float(correct), "acc": float(correct), "pred": answer or "INVALID"}
