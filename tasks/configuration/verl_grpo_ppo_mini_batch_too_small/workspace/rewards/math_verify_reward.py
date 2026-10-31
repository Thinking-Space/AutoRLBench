"""Reward: Math-Verify equivalence between the response's final answer and the gold answer.

Thin wrapper around verl's built-in scorer ``verl.utils.reward_score.math_verify.compute_score``
(the one verl's ``default_compute_score`` comments suggest for MATH-style data). Wired in through
``reward.custom_reward_function.path/name`` (see run_gsm8k_grpo.sh); verl calls
``compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)`` from its
reward loop and uses the same function for the held-out validation metric.
"""

from verl.utils.reward_score import math_verify


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if solution_str is None:
        return 0.0
    return float(math_verify.compute_score(solution_str, str(ground_truth)))
