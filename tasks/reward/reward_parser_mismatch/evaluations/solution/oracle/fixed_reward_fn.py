from verl.utils.reward_score.naive_dapo import _last_boxed_only_string, grade_answer

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    text = str(solution_str or "")
    truth = str(ground_truth)
    # The existing math scorer balances braces, including fractions and radicals.
    extracted = _last_boxed_only_string(text)
    if extracted is None:
        return {"score": 0.0, "acc": False, "parser_matched": False}
    correct, _ = grade_answer(extracted, truth)
    return {
        "score": 1.0 if correct else 0.0,
        "acc": bool(correct),
        "parser_matched": True,
    }
