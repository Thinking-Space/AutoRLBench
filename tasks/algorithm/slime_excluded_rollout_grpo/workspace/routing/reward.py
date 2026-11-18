"""Normalize terminal service-ticket rewards for the Slime actor."""
import json
import os
from pathlib import Path

from routing.reward_common import collect_rollouts, expand_rollout_values, task_advantages


def post_process_rewards(args, samples):
    records, groups = collect_rollouts(args, samples)
    values = {}
    for group in groups.values():
        values.update(task_advantages(args, group))
    with (Path(os.environ["ROUTING_OUTPUT"]) / "rollouts.jsonl").open("a") as stream:
        for record in records:
            sample = record.indexed_samples[-1][1]
            row = {"group": record.key[0], "sample": record.key[1], "status": sample.status.value,
                   "masked": sample.remove_sample, "reward": record.task_reward, "target": sample.label,
                   "response_tokens": sample.response_length, **record.metadata}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    return raw_rewards, expand_rollout_values(samples, records, values)
