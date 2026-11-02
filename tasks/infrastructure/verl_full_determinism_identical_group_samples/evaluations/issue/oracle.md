# Canonical issue oracle

Reconciled from the task's frozen evidence and a controlled comparison on the
workspace's verl checkout at `6f2aa24abfc01cc72811b376aeb3936503885096`
(unmodified): the defective arm is the released code, the repaired arm changes
only the per-request seed in
`verl/workers/rollout/vllm_rollout/vllm_async_server.py`. There is no upstream
fix; current verl main still has the same line.

Expected verdict: **unhealthy**.

## Root cause

- `verl/workers/rollout/vllm_rollout/vllm_async_server.py::vLLMHttpServer.generate`, full-determinism seed injection
  - kind: `symbol`
  - observed: with `actor_rollout_ref.rollout.full_determinism=True`,
    `generate` calls
    `sampling_params.setdefault("seed", self.replica_rank + self.config.seed)`,
    so every request served by a replica gets the same sampling seed (42 for
    the single replica here). Full determinism also runs vLLM in batch-invariant
    mode, so the `rollout.n=8` copies of a prompt have bitwise-identical logits
    regardless of how they are batched. Same distribution plus same random
    stream gives the same tokens: all 8 samples of every prompt are
    byte-identical. GRPO's group standard deviation is then 0 for every group,
    every advantage is exactly 0, the policy-gradient loss and gradient are 0,
    and the actor never moves away from the reference model.
  - expected: a seed that differs between the `n` samples of a prompt and is
    reproducible across restarts, for example derived from the sample's row in
    the batch (`priority`, assigned by `AgentLoopManager` with
    `np.arange(len(batch))` before dispatch) and the prompt tokens. The request
    id cannot serve: `LLMServerClient` sends vLLM a fresh `uuid4` per call.

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1`.
- Held-out accuracy is 0.08719 at every validation (steps 0, 5, 10, 15, 20,
  25 and 30), identical to the base model's greedy accuracy. The same recipe
  with a per-sample seed reaches 0.6543 at step 5 and 0.7384 at step 30.
- `critic/advantages/max`, `critic/advantages/min`, `actor/pg_loss`,
  `actor/grad_norm` and `actor/kl_loss` are exactly 0 at all 30 steps.
- Sampling is not greedy: rollouts use temperature 1.0 and `actor/entropy` is
  between 0.33 and 0.68. Training reward (`critic/score/mean`) varies between
  0.06 and 0.27 from batch to batch because the prompts differ, not because the
  policy changes.
- In the rollout dumps every prompt appears 8 times with 8 identical responses
  (64 of 64 groups at steps 1, 15 and 30 in `outputs/rollouts/`; 64 of 64 at
  all 30 steps in the full recorded dumps). With the repaired seed, 0 of 64
  groups are identical at any step.
- Frozen evidence is in `outputs/train.log`, `outputs/console-history.jsonl`,
  `outputs/wandb-history.jsonl`, `outputs/rollouts/` and `outputs/val_generations/`.

## Trigger conditions

- `actor_rollout_ref.rollout.full_determinism=True` (vLLM rollout), which both
  injects the replica-wide seed and enables batch-invariant kernels.
- `actor_rollout_ref.rollout.n > 1` with sampling (temperature > 0), as every
  group-relative estimator (GRPO here) requires. With `n=1`, as in verl's own
  determinism tests, the defect does not show.
- Copies of a prompt served by the same replica; with one replica every copy is.

## Documentation reconciliation

- verl's determinism documentation says each replica uses
  `replica_rank + seed`. That makes two runs reproducible, which they are: the
  defect is not nondeterminism but that every sample of a replica shares one
  random stream.
- Not greedy decoding: temperature is 1.0, entropy is far from 0, and
  validation (which is greedy by configuration) is unaffected.
- Not the uid grouping: the 8 rows of a prompt share one uid and form a group
  of 8, unlike a singleton-group defect where advantages are non-zero but
  unnormalized. Here advantages are exactly 0.
- Not the reward function, learning rate, KL coefficient or answer format: the
  reward varies across prompts, and the same configuration with a per-sample
  seed learns within 5 steps.
- Turning full determinism off would also produce diverse groups, but it
  discards the reproducibility the run asked for; the defect is the seed
  derivation, not the mode.
