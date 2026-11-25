"""Reward dispatch for the frozen PPO recipe."""
from scoring.math_dapo import compute_score as dapo_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source.startswith("train-math"):
        import json

        from scoring.math_verify import compute_score as verify_score

        answer = solution_str.rpartition("</think>")[2] if "</think>" in solution_str else ""
        return 1.0 if any(verify_score(answer, str(gold)) > 0 for gold in json.loads(ground_truth)) else -1.0
    if data_source in {"AIME24", "AIME25", "AMC23"}:
        from scoring.math_verify import compute_score as verify_score

        return verify_score(solution_str, str(ground_truth))
    if data_source == "math_dapo" or data_source.startswith("aime"):
        return dapo_score(solution_str, str(ground_truth), strict_box_verify=True)
    raise NotImplementedError(f"No verifier for {data_source=}")
