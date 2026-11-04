"""
Custom reward for the GuruMath smoke runs.

Grading reuses grade_answer from verl.utils.reward_score.naive_dapo so
on-policy scores stay comparable with the offline evaluator. Answer
extraction follows the QA answer-tag contract (see
verl.utils.reward_score.search_r1_like_qa_em) that the shared scoring
stack was built around.
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

    match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        solution_str,
        flags=re.DOTALL,
    )

    if match is None:
        return {
            "score": 0.0,
            "acc": False,
            "parser_matched": False,
        }

    extracted_answer = match.group(1).strip()

    correct, _ = grade_answer(
        extracted_answer,
        ground_truth,
    )

    return {
        "score": 1.0 if correct else 0.0,
        "acc": bool(correct),
        "parser_matched": True,
    }
