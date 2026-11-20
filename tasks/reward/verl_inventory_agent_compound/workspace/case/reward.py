"""Batch scoring for the arithmetic task."""

import re

import torch


def answer_matches(solution_str, ground_truth):
    matches = re.findall(r"<answer>\s*(-?\d+)\s*</answer>", solution_str)
    return float(len(matches) == 1 and int(matches[0]) == int(ground_truth))


def reward_callback(solution_strs, ground_truths):
    values = [answer_matches(text, truth) for text, truth in zip(solution_strs, ground_truths, strict=True)]
    return values


def collect_rewards(values, batch_size):
    if len(values) != batch_size:
        raise ValueError(f"Reward callback returned {len(values)} values for {batch_size} completions")
    rewards = torch.empty(batch_size, dtype=torch.float32)
    rewards[:] = torch.tensor(values, dtype=torch.float32)
    return rewards.tolist()


def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, **kwargs):
    if not (len(data_sources) == len(solution_strs) == len(ground_truths)):
        raise ValueError("Scoring input lengths must match")
    if not solution_strs:
        return []
    rewards = collect_rewards(reward_callback(solution_strs, ground_truths), len(solution_strs))
    return [
        {"score": reward, "acc": answer_matches(text, truth), "oracle_acc": answer_matches(text, truth)}
        for reward, text, truth in zip(rewards, solution_strs, ground_truths, strict=True)
    ]
