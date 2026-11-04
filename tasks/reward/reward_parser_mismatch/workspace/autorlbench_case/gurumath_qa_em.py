"""
GuruMath QA-EM grader for the math__* sources.

Adapted from verl.utils.reward_score.search_r1_like_qa_em with extraction
switched to the boxed-answer contract of the GuruMath prompts; grading keeps
grade_answer from naive_dapo for numeric/LaTeX equivalence. Wired as the
default scorer through recipe/vimpo/config/vimpo_trainer.yaml.
"""

import re

from verl.utils.reward_score.naive_dapo import grade_answer


def extract_boxed(solution_str):
    matches = list(re.finditer(r"\\boxed\{([^{}]+)\}", solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    solution_str = str(solution_str)
    ground_truth = str(ground_truth)

    extracted_answer = extract_boxed(solution_str)

    if extracted_answer is None:
        return {
            "score": 0.0,
            "acc": False,
            "parser_matched": False,
        }

    correct, _ = grade_answer(
        extracted_answer,
        ground_truth,
    )

    return {
        "score": 1.0 if correct else 0.0,
        "acc": bool(correct),
        "parser_matched": True,
    }
