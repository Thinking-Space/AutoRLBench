# Tool observations erase earlier SQL action returns

Qwen3-4B learns read-only warehouse SQL with REINFORCE++, but masked tool observations reset the return accumulator and remove terminal reward from earlier actions.

| | |
| --- | --- |
| **Author** | PurvaChiniya |
| **Domain** | algorithm |
| **Tags** | `verl` `tool-use` `issue-only` |
| **Model** | Qwen/Qwen3-4B |
| **Dataset** | Generated operations SQL: 512 train, 12 pilot development, 128 sealed cases |
| **Framework** | veRL, `run.sh` |
| **Environment** | `tool-rl-vllm024`; recorded on 2 × H100 80 GB, 24 CPUs, 128 GiB RAM |
| **Evaluations** | issue |
| **Recorded runs** | 1 defective + 1 control, seed 17, 3 optimizer updates; 389 s + 390 s including startup |

## Difficulty

The warehouse has one-to-many orders, items, refunds, and shipments. Tasks require net revenue, late-delivery fractions, or partial-fulfillment aggregation. Agents inspect schemas and execute bounded read-only queries, then submit SQL and result rows. Binary reward requires correct rows and semantically correct SQL on the original database and hidden deterministic perturbations. Repeated refunds, missing children, cancellations, and partial shipments make naive joins and hardcoded answers fail. Train/development/test wording templates and database seeds are disjoint. The generator supports 96 development cases; this recorded pilot uses 12. The 128 test cases were not evaluated with the policy.

In the defective return estimator, an internal zero in `response_mask` clears the running return. Such zeros are tool observations, not episode boundaries. Of 192 recorded training trajectories, four successful tool trajectories contain 2,480 earlier action tokens; every one receives raw return zero despite terminal reward one. Their whitened advantages are negative, approximately −0.0920 or −0.1041: **zero raw return does not mean zero advantage or zero gradient**.

Exact counterfactual replay of the same full recorded batches through the upstream parent and repaired functions reproduces the observed values and restores those earlier returns to one. An independently sampled paired GPU control has one successful tool trajectory with 587 earlier action tokens, all receiving return one. Same-seed asynchronous runs do not generate identical trajectories. Both arms have zero development accuracy at steps 0–3; task difficulty remains a confound for learning performance. The evidence establishes the credit-assignment defect, not a measured accuracy gain.

## Reference solution

**Issue:** trace terminal reward through the tool-observation gaps in `compute_reinforce_plus_plus_outcome_advantage`, identify the incorrect accumulator reset, and distinguish policy-loss masking from reward propagation. The upstream repair preserves accumulated reward across zero-mask positions; tool observations remain excluded from the policy loss. See [repair.patch](reference/repair.patch) and [the identical-batch replay](reference/credit_replay.json).

Source snapshots are pinned to the upstream defect and repair. Both GPU arms use the same compatibility adaptation: veRL's existing pure-Torch padding helpers support SDPA without the optional compiled flash-attention extension. This adaptation does not change return computation. Prompt wording and operational correctness reminders were tuned on training/development pilots before the final pair; reward remained strict binary and sealed test performance was not consulted.

## Verification

**Issue:** the shared LLM judge compares `agent_answer.md` with the hidden oracle, weighting verdict/root cause/evidence 0.2/0.6/0.2. The canonical verifier requires verdict score 1.0 and both root-cause and evidence scores strictly greater than 0.8 to pass. No solution tier or calibrated performance threshold is claimed. The proposed three-seed, 60-update campaign was not run, and the short control does not exceed its step-zero development score.

The package includes all per-step metrics, complete reward/mask/return/advantage batches, tool traces, and rollout evidence. Raw W&B binaries are preserved outside the agent workspace under `reference/raw_wandb`. See [verification.md](reference/verification.md) for CPU, image, preparation, and structural checks.

One complete debugging-agent trial ran on November 4, 2026: Harbor Terminus-2 with GPT-5.4, graded by the canonical GPT-5.2 judge. **The evaluation completed; the agent failed with 20/100.** It correctly called the run unhealthy but blamed SQL query errors and missed the return-masking bug. All 504 original non-model code/evidence files remained unchanged. This diagnosis run used CPUs and took about five minutes including setup and grading. See the [recorded answer, judge result, and run settings](reference/agent-evaluation-01/README.md).
