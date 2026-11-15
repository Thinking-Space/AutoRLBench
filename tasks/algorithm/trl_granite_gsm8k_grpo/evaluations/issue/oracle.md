# Canonical issue oracle

Expected verdict: **unhealthy**. The run improves, but far less than the same
recipe without the defect; "the run is healthy because accuracy rises" is a
wrong verdict.

## Root cause

### `trl/trainer/grpo_trainer.py::GRPOTrainer._prepare_inputs` — the buffer is indexed by the optimizer step

Each optimizer step generates and scores one batch of 64 rollouts (8 prompts x
8 completions), shuffles it and splits it into `steps_per_generation = 16`
micro-batches of 4, one per gradient-accumulation micro-step. The micro-batch
for the current micro-step is selected with

```python
inputs = self._buffered_inputs[self.state.global_step % self.args.steps_per_generation]
```

`self.state.global_step` counts optimizer steps and is constant across the 16
micro-steps of one update; the micro-step counter is `self._step`, which also
drives regeneration (`self._step % generate_every`). All 16 accumulation
micro-steps therefore train on the same 4 rollouts, and the other 60 scored
rollouts of every generation batch never reach the loss.

- Observed: one 4-rollout micro-batch is accumulated 16 times per optimizer
  step; the effective policy-gradient batch is 4 completions instead of 64.
- Expected: `self._buffered_inputs[self._step % self.args.steps_per_generation]`,
  so the 16 micro-steps consume the 16 distinct micro-batches.

Rewards, group statistics and advantages are computed on all 64 rollouts and
are correct, and the 4 trained completions carry their own advantages, so the
gradient direction is unbiased. The update is a 4-sample estimate instead of a
64-sample one.

## Observable impact

- Held-out greedy accuracy on all 1319 GSM8K test questions: 58.7% at step 0,
  78.4% at step 20 and 79.8% at step 40. The matched run that differs only
  in this index reaches 79.0% and 82.8% from the same step-0 model and the
  same first rollouts.
- The improvement is almost entirely answer formatting: among answers in the
  parseable `#### <number>` form, accuracy goes 0.799 -> 0.803 -> 0.807
  (matched run 0.799 -> 0.810 -> 0.837), while the format rate rises
  73.5% -> 98.8%.
- The logged loss is far from zero and changes sign (mean |loss| 0.210, up to
  0.697; matched run 0.0000). With one update per batch the policy term is the
  negated mean advantage of the trained completions, which averages to zero
  over a full set of GRPO groups but not over 4 completions.
- Gradient norms are large and erratic (median 6.04, max 29.9; matched run
  median 1.65, max 3.5), and 6 of 40 steps have a gradient norm below 0.05,
  consistent with the 4 trained completions coming from zero-variance groups
  while other groups in the batch had reward variance.
- Nothing fails: the run completes, metrics are finite, and every logged
  rollout statistic (`reward`, `reward_std`, `frac_reward_zero_std`,
  completion lengths) is computed over all 64 rollouts, most of which the
  optimizer never sees.

## Wrong attributions

- "Healthy, accuracy improves": the improvement is a fraction of what the same
  recipe reaches.
- Learning rate 1e-6, beta 0.001, 8 generations, the 512-token limit, the
  constant scheduler or gradient clipping: identical in the matched run.
- Reward parsing, answer format, GSM8K labels or data order: unchanged between
  the runs; the rollout dumps carry correct rewards.
- Groups with zero reward variance, or a statement that GRPO is inherently
  sample-inefficient: both runs see the same rollouts.
- Advantage computation or advantage/completion alignment: the trained
  completions carry their own correct advantages.

A diagnosis passes when it names the buffered micro-batch indexing in
`_prepare_inputs` (optimizer step used where the micro-step counter belongs,
so one 4-rollout micro-batch is reused for all accumulation micro-steps) and
connects it to the reduced held-out improvement.
