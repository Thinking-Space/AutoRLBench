"""
GSM8K-format reward for the offline QA pipeline.

Grades the final `#### <answer>` line of a response against the ground truth,
reusing grade_answer from verl.utils.reward_score.naive_dapo for the
non-numeric fallback so scores stay comparable with the other autorlbench_case
scorers. Not wired into the GuruMath smoke runs (see README).
"""

import re

from verl.utils.reward_score.naive_dapo import grade_answer


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    solution_str = str(solution_str)
    ground_truth = str(ground_truth)

    match = None
    for match in re.finditer(r"####\s*(-?[\d,.]+)", solution_str):
        pass

    if match is None:
        return {
            "score": 0.0,
            "acc": False,
            "parser_matched": False,
        }

    extracted_answer = match.group(1).replace(",", "")

    correct, _ = grade_answer(
        extracted_answer,
        ground_truth,
    )

    return {
        "score": 1.0 if correct else 0.0,
        "acc": bool(correct),
        "parser_matched": True,
    }
