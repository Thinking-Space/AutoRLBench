"""Terminal GSM8K reward dispatched by verl's NaiveRewardManager."""
from observer import emit, exact


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    score = exact(solution_str, ground_truth)
    emit('rewards', {'response': solution_str, 'ground_truth': str(ground_truth),
         'canonical_answer': str(extra_info['canonical_answer']), 'score': score,
         'exact': exact(solution_str, extra_info['canonical_answer']),
         'index': int(extra_info['index']), 'split': extra_info['split']})
    return score
