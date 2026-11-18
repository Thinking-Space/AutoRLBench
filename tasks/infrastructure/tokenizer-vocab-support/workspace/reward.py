"""Match JustRL's separate training and online evaluation verifiers."""
from verl.utils.reward_score.math_dapo import compute_score as dapo_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source.startswith('train-math'):
        # Skywork-OR1 RL data: ground truth is a JSON list of accepted answers. Only a finished thought
        # (</think> present) is scored, so numbers inside truncated reasoning cannot earn reward.
        import json

        from verl.utils.reward_score.math_verify import compute_score as verify_score

        answer = solution_str.rpartition('</think>')[2] if '</think>' in solution_str else ''
        return 1.0 if any(verify_score(answer, str(gold)) > 0 for gold in json.loads(ground_truth)) else -1.0
    if data_source in {'AIME24', 'AIME25', 'AMC23'}:
        from verl.utils.reward_score.math_verify import compute_score as verify_score

        # JustRL's online evaluation returns 0/1 and accepts boxed gold answers.
        return verify_score(solution_str, str(ground_truth))
    if data_source == 'math_dapo' or data_source.startswith('aime'):
        # Preserve JustRL's exact training reward: last 100 characters, +1/-1.
        return dapo_score(solution_str, str(ground_truth), strict_box_verify=True)
    raise NotImplementedError(f'No JustRL verifier for {data_source=}')
