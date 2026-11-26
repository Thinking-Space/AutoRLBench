from verl.utils.reward_score.math_dapo import compute_score


def score(data_source, solution_str, ground_truth, extra_info):
    return float(compute_score(solution_str, ground_truth, strict_box_verify=True)["score"] > 0)
