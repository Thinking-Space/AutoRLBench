"""Use verl's standard GSM8K final-answer reward."""
from verl.utils.reward_score.gsm8k import compute_score as gsm8k_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return gsm8k_score(solution_str, str(ground_truth), method="strict")
