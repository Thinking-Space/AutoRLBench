"""Route verification and reward aggregation for the training experiment."""
from reward_common import collect_rollouts, expand_rollout_values, task_advantages


def terminal_reward(ticket, answer, catalog_response):
    if answer is None:
        return 0.0
    return 1.0 if catalog_response and ticket.is_correct(answer) else 0.6


def post_process_rewards(args, samples):
    records, groups = collect_rollouts(args, samples)
    values = {}
    for group in groups.values():
        values.update(task_advantages(args, group))
    return expand_rollout_values(samples, records, values)
