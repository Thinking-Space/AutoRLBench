"""
Controlled AutoRLBench candidate bug.

The training prompt requires:
    \boxed{...}

but this scorer incorrectly expects:
    <answer>...</answer>

The mismatch is silent: a missing <answer> tag receives score 0
instead of raising an exception.
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

    # BUG:
    # Training prompts require \boxed{...}, but this parser was
    # accidentally changed to an XML-style answer contract.
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
