# Canonical issue oracle

Expected verdict **unhealthy GRPO learning signal**. The training completes and both the historical callback and the control finish with 144/144 verified routes. The diagnosis concerns the learning signal under tool interruptions.

## Root cause

`workspace/train.py::Rollout.fragments` encodes a timed-out catalog call as one terminal record with reward zero, status `aborted`, `remove_sample=True`, and no trainable decision. `workspace/reward_callback.py::post_process_rewards` groups all terminal records and forwards them to `task_advantages` before `update` selects trainable fragments. `workspace/reward_common.py` already collapses decision fragments to one reward per terminal rollout and uses the group mean and sample standard deviation. It receives a normalization population that includes already-excluded terminal records. Late masking removes the timeout from the loss while its zero still changes the completed rollouts' advantages.

Compute eligibility per terminal rollout before normalization, using exclusion flags rather than raw reward. A genuine scored zero should count. Expand the computed advantages back to the two decision fragments of each completed rollout and give excluded fragments no loss. A singleton eligible group and a group with equal eligible rewards should have zero centered advantage. An all-excluded group should never be normalized.

## Observable impact

- `workspace/evidence/rollouts-001.jsonl`, group 2, contains rewards `[0.6, 1.0, 1.0, 1.0, 0.0, 0.0]` of which the last two are timeouts. The complete 0.6 rollout gets advantage 0.0 in the historical computation and about −1.5 with the two timeouts removed.
- `workspace/evidence/rollouts-036.jsonl`, group 0, contains four completed rewards of 1.0 and two timeout zeros. Each completed rollout gets advantage about +0.6455 under the historical callback and 0 under eligible-only centering. Group identities and rollout outcomes match in the paired arms at this step.
- At round 36 the defective run logs 29 verified completions, seven timeouts, 58 trainable actions and a nonzero gradient norm of about 0.0007186. The control logs the same rollout counts and zero gradient norm. The interruption appears in the learning signal even once the policy has saturated.
- A passing diagnosis explains how a discarded sample can change its valid peers' advantages, using an actual group or connecting the last-round gradient to the complete equal-reward trajectories and masked timeouts. Mentioning timeouts or masking without connecting the peer effect merits partial root-cause credit.

## Incorrect attributions

The timeout is a trigger. The algorithmic defect is the use of timeout rewards in normalization. The training includes real policy updates and reaches full held-out accuracy. Fragment counting, the standard-deviation convention, and insufficient optimization do not explain the effect. Removing every zero-valued reward would also remove legitimate scored completions.
