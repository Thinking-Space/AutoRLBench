"""GSM8K reward: 1.0 when the last number in the response equals the reference answer."""
from verl.utils.reward_score import gsm8k


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return gsm8k.compute_score(solution_str, ground_truth, method="flexible")
