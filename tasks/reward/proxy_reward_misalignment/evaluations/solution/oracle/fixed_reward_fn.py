from verl.utils.reward_score import default_compute_score

TRAIN_SOURCE = "autorlbench_type3_proxy_train_v2"


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    score_source = (
        "math__math"
        if data_source == TRAIN_SOURCE
        else data_source
    )

    return default_compute_score(
        data_source=score_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
