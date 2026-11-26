"""Route the preserved Skywork math labels to VeRL's standard math verifier."""

import json

from verl.utils.reward_score.math_dapo import compute_score as math_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if not data_source.startswith("train-math-"):
        raise ValueError(f"Expected a Skywork math row, got {data_source!r}")
    answers = json.loads(ground_truth)
    if not isinstance(answers, list) or not answers or not all(isinstance(answer, str) for answer in answers):
        raise ValueError(f"Expected nonempty Skywork answer list, got {ground_truth!r}")
    for answer in answers:
        result = math_score(solution_str, answer)
        if result["acc"]:
            return result
    return result
