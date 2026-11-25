"""Arithmetic of the defect on one synthetic prompt group, using the workspace's own verl code on CPU.

Usage (from the workspace root, so that `verl` is importable):  PYTHONPATH=. python3 group_std_demo.py

Eight wrong answers (accuracy 0 for all). Each sample's reward is 0 minus 0.001 times its summed
log-ratio to the reference model, which is what apply_kl_penalty leaves in the token-level rewards.
The summed log-ratios below are made up; they span a few nats, as in the recorded run by step 20.
"""
import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

log_ratio_sum = torch.tensor([0.4, 1.9, -0.3, 2.6, 1.1, 0.2, 3.3, 0.8])  # sum_t (log pi - log pi_ref), nats
width = 4
mask = torch.ones(8, width)
index = np.zeros(8, dtype=int)
for coef in (0.001, 0.0001):
    rewards = torch.zeros(8, width)
    rewards[:, -1] = 0.0 - coef * log_ratio_sum
    print(f"KL coefficient {coef}: per-sample reward = {[round(x, 6) for x in rewards[:, -1].tolist()]}")
    for flag in (True, False):
        adv, _ = compute_grpo_outcome_advantage(
            token_level_rewards=rewards.clone(), response_mask=mask, index=index, norm_adv_by_std_in_grpo=flag
        )
        print(f"  norm_adv_by_std_in_grpo={flag!s:5}: advantages = {[round(x, 4) for x in adv[:, 0].tolist()]}")
