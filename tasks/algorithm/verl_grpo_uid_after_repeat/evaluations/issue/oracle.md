# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the staged
workspace and upstream verl `6f2aa24abfc01cc72811b376aeb3936503885096`.

Expected verdict: **unhealthy**.

## Root cause

- `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit`
  - kind: `symbol`
  - observed: the `batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)` assignment was moved to **after** `batch = batch.repeat(repeat_times=rollout.n, interleave=True)` and `batch.union(gen_batch_output)`. The batch at that point holds one row per sampled response, so every one of the `train_batch_size * rollout_n` rows receives its own fresh UUID.
  - expected: the uid is assigned **before** `self._get_gen_batch(batch)`, while the batch still holds one row per prompt, so the later `repeat(interleave=True)` duplicates each uid `rollout_n` times and all samples drawn from one prompt share a group key.

The uid is the grouping key that `compute_advantage` passes to
`compute_grpo_outcome_advantage` as `index` (`ray_trainer.py:243`,
`ray_trainer.py:256-257`). One uid per row makes every GRPO group a singleton.
`verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage` handles
`len(id2score[idx]) == 1` by silently substituting `id2mean = 0.0` and
`id2std = 1.0` instead of raising, so the group-relative baseline is never
subtracted and the per-group standard deviation never divides. The advantage
of each response therefore degenerates to its own raw outcome reward:
GRPO becomes REINFORCE with no baseline and no variance normalisation.

## Observable impact

Measured on the recorded 60-step defect run (`workspace/logs/`), whose
per-step metrics are in `metrics_<timestamp>.jsonl`.

- Primary source evaluation metric: `val-core/math_dapo/acc/mean@1`.
- **`critic/advantages/min` is exactly `0.0` at every one of the 60 steps.**
  No sampled response is ever assigned a negative advantage. A healthy GRPO
  batch must contain negative advantages whenever a group mixes correct and
  incorrect responses, because the group mean is subtracted.
- **`critic/advantages/max` is exactly `0.9999990463256836` at every step
  where any response scored**, and `0.0` at the steps where none did. That
  constant is fp32 `1 / (1 + 1e-6)`: it is the raw reward `1.0` divided by the
  substituted `id2std = 1.0` plus the `epsilon = 1e-6` guard in
  `compute_grpo_outcome_advantage`. A real group standard deviation would
  divide by something else at essentially every step, so a max that is
  bit-identical across every rewarded step is a direct fingerprint of the
  singleton-group branch.
- `critic/advantages/mean` is strictly positive on every rewarded step
  (`0.0048` to `0.0674` in the recorded run) instead of sitting near zero.
  Note it is a response-token-weighted mean, not the plain mean of the
  sequence rewards, so it is close to but not exactly equal to
  `critic/rewards/mean` — comparing the two for exact equality is not the
  right check; the sign and the absence of cancellation are.
- `critic/returns/mean` equals `critic/advantages/mean` to within `1e-9` at
  every step, because the GRPO estimator returns the advantage tensor for
  both. This is also true of a healthy run and is not by itself evidence.
- `actor/pg_loss` never becomes positive: it is `0.0` on unrewarded steps and
  negative on every rewarded step, because every token carries a
  non-negative advantage, so nothing ever pushes a sampled response down. A
  healthy run's `pg_loss` fluctuates around zero taking both signs, at a
  comparable magnitude — the separator is the sign, not the order of
  magnitude (both arms sit at order `1e-3` to `1e-2` on rewarded steps).
- Held-out `val-core/math_dapo/acc/mean@1`: `0.0454` (step 0) → `0.0437`
  (20) → `0.0476` (40) → `0.0570` (60). The curve is **not** collapsed, and
  at a 60-step budget it does not on its own separate this defect from a
  healthy short run. The advantage statistics do. A diagnosis resting mainly
  on the held-out curve has not identified the defect.

## Trigger conditions

- `algorithm.adv_estimator=grpo` (or any estimator that reads
  `non_tensor_batch["uid"]` as its grouping index). The defect is inert for
  estimators that do not group, such as GAE.
- `actor_rollout_ref.rollout.n > 1`. At `rollout.n == 1` the moved assignment
  is behaviourally equivalent to the baseline, because every group is a
  singleton either way.
- The run does not crash, warn, or change tensor shapes: the silent
  singleton-group branch in `compute_grpo_outcome_advantage` absorbs the
  defect, so the only evidence is the numeric signature of the advantages.

## Documentation reconciliation

- `ray_trainer.py:1510` still carries the upstream comment "this usually
  changes the order of data in the `batch`, which won't affect the advantage
  calculation (since it's based on uid)". That comment is true of the baseline
  and remains in the staged tree; it describes the invariant the moved
  assignment breaks, and is not itself a defect.
- The `# add uid to batch` comment moved with the assignment, so the staged
  tree has exactly one uid assignment in the training loop, as upstream does.
  Counting assignments does not reveal the defect; their position relative to
  `repeat` does.
- A low absolute held-out accuracy is a symptom, not the root cause. The
  recipe's base accuracy on DAPO-Math is only a few percent, so the held-out
  curve alone does not separate this defect from a healthy short run. The
  advantage statistics do.
- Reporting `train_batch_size`, `rollout.n`, the learning rate, the reward
  function, or the absence of KL loss as the root cause is a wrong attribution:
  all of them match the healthy recipe for this task.
- Reporting "GRPO group size is 1" or "advantages are not normalised" without
  tying it to where the uid is assigned relative to `batch.repeat` is an
  incomplete attribution and must not receive full `root_cause_score`.
