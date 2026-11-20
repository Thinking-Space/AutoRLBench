# Canonical issue oracle

Expected verdict **unhealthy GRPO learning signal under tool interruptions**. The optimizer ran and the policy initially improved on the held-out tasks. The evidence establishes incorrect credit assignment for completed rollouts.

## Root cause

`workspace/routing/rollout.py::generate` handles a catalog timeout by returning a record with zero reward, `Sample.Status.ABORTED`, `remove_sample=True`, and an all-zero response loss mask. The training callback in `workspace/routing/reward.py::post_process_rewards` sends the full list of terminal records for each prompt to `task_advantages`. The helper in `workspace/routing/reward_common.py` groups by prompt and rollout identity and computes a GRPO mean and sample standard deviation over the records it receives. The normalization population already includes interrupted records when `workspace/slime/slime/ray/rollout.py::_convert_samples_to_train_data` applies the loss masks.

This order makes a timeout affect eligible peer rewards even though its tokens contribute no policy loss. Compute group-relative advantages over eligible terminal records, using their exclusion status. Preserve legitimate zero scores. No score should enter the group statistics more than once for a terminal trajectory.

## Observable impact

- `workspace/rollouts/rollout-000.jsonl`, group 1, records four terminal rewards `[0, 1, 0.6, 0.6]`. The first is a masked catalog timeout. The recorded callback gives the 0.6-reward completions about **+0.1213**. Excluding that timeout gives about **−0.5773** for the same completions. This is a comparison on the same sampled trajectories.
- `workspace/logs/training.log` records the completed optimization and held-out evaluation. The run reached 16/16 before finishing at 0/16. The decline alone cannot identify its cause.
- A passing diagnosis connects the eligibility omission to changed peer advantages and anchors the claim in a recorded reward group or the native training signal. A timeout report or a reference to the loss mask without the peer effect merits partial root-cause credit.
