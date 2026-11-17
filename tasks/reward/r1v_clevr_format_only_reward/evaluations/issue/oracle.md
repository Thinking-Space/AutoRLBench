# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

The reward never checks the answer. There is one root cause:

- **Path:** `workspace/run_grpo.sh`, line 14 (`REWARDS="${REWARD_FUNCS:-sformat}"`),
  passed to the trainer as `--reward_funcs $REWARDS` on line 29.
- **Observed:** the only reward function is `sformat`, which
  `open_r1/grpo.py` maps to `strict_format_reward`. It returns 1.0 when the
  whole completion matches `<think>...</think><answer>integer</answer>` and 0.0
  otherwise. The ground-truth count (`solution`) is not an argument of this
  function, so a completion with the wrong number earns the same reward as one
  with the right number.
- **Expected:** the reward includes a term that compares the answer with the
  ground truth. `open_r1/grpo.py` already registers two such functions,
  `count` (`count_reward`, exact integer match) and `accuracy`
  (`accuracy_reward`); neither is passed by the launch script.

Causal chain: GRPO advantages are computed from this reward alone, so the
policy is optimized only toward the output shape. The format reward saturates
within the first 16 steps, after which most of the remaining gradient signal is
noise with respect to counting, and held-out accuracy is free to drift. It
does: it rises while the format is still being learned, then falls back to
near its starting value.

## Observable impact

From `workspace/results/train.log` (150/150 steps, one log line per step),
`workspace/results/eval_metrics.jsonl` and `workspace/results/eval_outputs/`:

- `rewards/strict_format_reward` is the only reward column in the log; there is
  no accuracy or count column. It averages 0.715 over steps 1-15 and 0.972 over
  steps 16-150 (minimum 0.875). In 53 of the last 135 steps every
  sampled completion earns 1.0, so `reward_std` is 0 and the advantages are all
  zero.
- Held-out CLEVR accuracy (`split: val`, 500 items, greedy decoding):
  0.602 at step 0, 0.744 at step 25, 0.828 at step 50, 0.842 at step 75,
  0.584 at step 100, 0.700 at step 125, 0.656 at step 150. The last value is
  5.4 points above the untrained model and 18.6 points below the step-75 peak.
- Out-of-distribution accuracy (`split: ood_val`, 1000 Super-CLEVR items):
  0.103 at step 0, 0.243 at step 25, then 0.199, 0.221, 0.192, 0.198, and
  0.187 at step 150.
- Training reward and held-out accuracy disagree: between step 75 and step 100
  the training reward stays at 0.97 while held-out accuracy falls from 0.842 to
  0.584.

## Symptoms, not root causes

- At step 100, 139 of 500 held-out outputs never reach `</answer>`: the model
  enumerates objects in a repeating list until the 256-token limit (6 such
  outputs at step 75, 76 at step 150). This is how the drift shows up under
  greedy decoding. It is a consequence of a reward that does not anchor the
  content of the reasoning, not a decoding or token-limit defect: the same
  limit and decoding give 6 truncated outputs at step 75.
- Among the outputs that do parse, the share of correct answers also falls,
  from 0.852 at step 75 to 0.774 at step 150.
- `completion_length` grows from about 57 tokens (steps 16-45) to about 78
  tokens (steps 91-150).
- `kl`, averaged over 15-step blocks, stays between 0.02 and 0.06.

## Not the root cause

- **"Training did not learn" / "the reward is not improving".** The reward
  improves and saturates. The run optimizes what it was told to optimize.
- **Learning rate (`1e-6`), KL coefficient (`--beta 0.04`), group size
  (`--num_generations 8`), batch shape, or the 150-step budget.** These are
  ordinary values for this trainer. No value of them makes a format-only reward
  carry information about the count.
- **`--max_completion_length 256` or the evaluation's 256 new tokens.**
  Truncation is a symptom (see above).
- **Bugs in `strict_format_reward`, `count_reward` or `accuracy_reward`.** Each
  does what its name says. The defect is which of them the launch script
  selects.
- **The evaluation script.** `eval_checkpoints.py` uses the same regular
  expression, the same 500 and 1000 items and greedy decoding at every step,
  including step 0.
- **The training data or the train / out-of-distribution gap.** The
  out-of-distribution set is harder (0.103 at step 0), but the in-distribution
  held-out set shows the same rise and fall, so distribution shift does not
  explain it.
- **Over-training / missing early stopping.** Stopping at step 75 would give a
  better checkpoint, but only because the format had been learned by then.
  Nothing in the objective holds accuracy at that level.

## Evidence

A passing diagnosis ties the launch script's reward selection to the reward
code and to the divergence between training reward and held-out accuracy. It
should cite at least: (1) `--reward_funcs sformat` in `run_grpo.sh` and the
fact that `strict_format_reward` ignores `solution`; (2) the saturated
`rewards/strict_format_reward` in `train.log` with no accuracy column; (3) the
held-out accuracy series in `eval_metrics.jsonl`, in particular that the final
value is close to step 0 and well below the mid-run peak.

Naming a reward problem in general terms ("reward hacking", "the reward is
too easy") without stating that the answer's correctness is not rewarded earns
partial root-cause credit (0.3) and cannot pass.
