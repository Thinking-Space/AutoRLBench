# Excluded OpenClaw rollouts contaminate GRPO advantages

A complete tool-routing GRPO run reproduces how the historical OpenClaw reward callback gives identical successful completions positive advantages when their group includes interrupted rollouts.

| | |
| --- | --- |
| **Author** | Pengcheng Huang |
| **Affiliation** | Northeastern University |
| **Type** | Algorithm |
| **Domain** | Tool-use RL |
| **Tags** | `grpo`, `reward-normalization`, `aborted-rollouts`, `agent-rl` |
| **Framework** | PyTorch and Transformers with the historical Slime reward helpers |
| **Run time** | 218.3 seconds in the recorded run, including training, evaluation and saves after model load |
| **Resources** | One NVIDIA H800 GPU |

## Difficulty

Training runs to completion and held-out accuracy saturates. The defect is in the learning signal during groups with tool timeouts. The workspace starts from public Qwen3.5-0.8B and runs 36 rounds of on-policy sampling, two clipped GRPO optimizer updates per round, and periodic checkpointing. The policy samples a tool choice and a final route from the available action tokens. A local catalog supplies observations. The simulator generates 2,048 training tickets and 48 held-out tickets, with three sampled attempts per held-out ticket and timeouts disabled for evaluation.

This is a reconstruction of the historical task-reward path using a public checkpoint, a local tool environment, and scheduled catalog timeouts. The trainer exercises the task-reward component of the original objective. The two arms use the same source, seed, schedule, checkpoint and update budget, with the eligibility change confined to `reward_callback.py`. Actual timeouts depend on whether a sampled rollout calls the catalog. The initial evaluation and the first and final recorded rollout batches match across the two arms.

| Evidence | Historical callback | Eligibility control |
| --- | ---: | ---: |
| Step 0 verified held-out routes | 42/144 | 42/144 |
| Step 6 verified held-out routes | 141/144 | 144/144 |
| Step 12 verified held-out routes | 144/144 | 142/144 |
| Step 36 verified held-out routes | 144/144 | 144/144 |
| Step 36 completed rollouts with reward 1.0 | 29/29 | 29/29 |
| Step 36 gradient norm, averaged over the two updates | 0.0007186256 | 0.0 |

In the step 1 trace, group 2 has rewards `[0.6, 1, 1, 1, 0, 0]`. The two zeros come from timed-out calls. Including them puts the 0.6 rollout at the group mean, suppressing its negative advantage of about −1.5. In the final trace, group 0 contains four successful completions and two timeouts. All four eligible rewards are 1.0. Including the interrupted runs gives those completions advantages of about +0.6455. Filtering the interrupted records returns zero for each.

## Reference solution

- **issue** — Identify how `workspace/reward_callback.py::post_process_rewards` sends interrupted records to `task_advantages` before the late loss selection in `workspace/train.py::update`. Normalize the eligible terminal records before broadcasting back to decision fragments. A scored zero from a legitimate completion would remain eligible.

The task is issue-only because the paired run identifies a gradient defect and both arms end at the same held-out score.

## Verification

- **issue** — Inspect the full per-round training and held-out evaluation log in `workspace/evidence/metrics.jsonl` and the decision traces at rounds 1, 18 and 36. Recompute the affected group statistics from the raw rewards and confirm that the late mask drops the interrupted trajectories after their rewards have contributed to normalization. The original training and paired control each completed 72 optimizer updates with valid saved adapters on the declared shared image. Prepare the pinned public checkpoint with `environment/prepare.sh`, then run `python3 train.py --out outputs` from the workspace on one compatible GPU to repeat the training.
