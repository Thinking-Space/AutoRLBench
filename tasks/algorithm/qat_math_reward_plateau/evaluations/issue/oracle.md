# Canonical issue oracle

Expected verdict is **unhealthy**. The recorded run loses sampler-policy
correction in the actor loss and discards valid positives in the training
reward before computing advantages.

## PPO score substitution

The quantized vLLM sampler emits a chosen-token score.
`qat/vendor/verl/verl/trainer/ppo/rollout_corr_helper.py` copies it into
`batch.batch["old_log_probs"]` in the enabled rollout bypass. The recorded
configuration has one minibatch, one PPO epoch, and
`use_rollout_log_probs=false`. In
`qat/vendor/verl/verl/workers/actor/dp_actor.py` these settings enter the
`on_policy` branch and replace the batch score with
`log_prob_for_loss.detach()`.

The ratio `exp(log_prob_for_loss - old_log_prob)` has value one because the
scores have equal numerical values. The numerator retains its gradient, so
the actor can still learn through ordinary policy gradients. Both training
logs show zero actor PPO KL and clipping across all 30 steps, including 27
with nonzero gradient. At step 2 the gradient norm is about 0.548 at a
learning rate of `3e-7`. The independent `qat/rollout_logprob_mae` is about
0.363 at step 1. The two identical bypass perplexities compare the rollout
score tensor with itself.

## Training answer extraction

The native custom reward path loads the adapter staged as `math_reward.py`.
It decodes Skywork's answer list and calls
`qat/vendor/verl/verl/utils/reward_score/math_dapo.py` with the default
`strict_box_verify=false`. The scorer cuts the input to the last 300
characters and invokes the Minerva extractor, which looks for an `Answer:`
field. A correctly formatted LaTeX box alone can yield `[INVALID]` or leave
a different string as the prediction. The scorer records `score=-1` and
`acc=false` on those false negatives.

For example, line 2 of
`evidence/training/steps-01-08/rollouts/4.gen_001.jsonl` ends with
`\boxed{32}` against the matching Skywork label `["32"]`. It has
`pred="[INVALID]"`, `acc=false`, and `score=-1.0`. All 64 scores at step 4
are negative and the actor gradient is zero there. Accept an equivalent
example established from the packaged labels and rollouts. A verifier for
this dataset should also accept a matching final boxed answer.
Sparse or constant reward affects the advantages. The unit PPO ratio at
steps with active policy gradients follows from the score substitution.

## Held-out evidence and attribution

The AIME24 records contain 31/240 at E0, 41/240 at CP8, and 41/240 at CP30,
with one capped response per snapshot. The learning followed by a plateau
is consistent with both bugs and actual updates. The inherited
`QAT_PROBE_OBJECTIVE=kl` computes and logs block alignment while the
backward pass uses `pg_loss` alone. The PPO ratio has already been formed
when this choice is made. The zero learning rate at step 1 and the
all-negative reward batches have different effects. The logs show native
CP8 restore and training through CP30. A post-save worker warning cannot
explain the preceding training metrics.

## Grading

- Full credit requires an unhealthy verdict, both concrete causal paths
  with solver-visible evidence, and the 31/240 to 41/240 to 41/240
  trajectory with an explanation of why the actor can still update. The
  PPO explanation should trace the actual tensor replacement. The scorer
  explanation should identify the actual extractor and give a documented
  boxed-answer miss.
- Give partial credit for either correctly traced defect with evidence, or
  for tracing both with a material evidence gap. The PPO defect disables
  intended clipping throughout the run. The scorer defect removes valid
  reward signal, including on some all-negative batches. A generic account
  of zero KL, quantization, truncation, reward sparsity, or the plateau
  leaves the implementation defects unexplained.
