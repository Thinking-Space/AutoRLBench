# Reward for the GSM8K + MATH mixture.
# GSM8K keeps verl's strict "####" scorer; MATH-style sources use Math-Verify for symbolic equivalence
# (e.g. \frac{1}{2} == 0.5), which the regex-based math_reward scorer misses.
from verl.utils.reward_score import gsm8k, math_verify


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == "openai/gsm8k":
        return gsm8k.compute_score(solution_str, ground_truth)
    return math_verify.compute_score(solution_str, ground_truth)
