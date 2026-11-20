# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

The training set cannot teach what the out-of-distribution held-out set
measures, and training on it past the first 100 steps removes the ability the
base model had. Both parts are essential:

1. **Data.** `workspace/data/train` has 68,000 rows and every row asks the same
   question, "How many items are there in the image?". Its answers cover only
   the integers 3 to 10, about 8,500 rows each. `workspace/data/ood_val`
   (1,000 Super-CLEVR rows) asks 995 distinct questions, such as "How many
   other objects are the same shape as the small gray rubber object?", and 864
   of its answers are 0, 1 or 2: values that never occur as a training label.
   `workspace/data/val` has the same single question and the same 3-10 answer
   range as the training set.
2. **Causal chain to the failure.** With one question in every prompt, the
   question text carries no information about the reward. The exact-match
   reward (`count_reward`) therefore rewards one behaviour only: output the
   total number of objects in the image. GRPO moves the policy toward that
   behaviour for every prompt. On the out-of-distribution questions the policy
   increasingly ignores what is asked and returns a total count in the 3-10
   range, which is wrong for the 864 items whose answer is 0-2.

Naming only "train / evaluation distribution shift" or only "over-training",
without the single question and the answer range and without the
question-ignoring behaviour, earns partial root-cause credit (0.3) and cannot
pass.

## Observable impact

From `workspace/results/eval_metrics.jsonl`, `workspace/results/eval_outputs/`
and `workspace/results/train.log` (400/400 steps):

- Out-of-distribution accuracy (`split: ood_val`, 1,000 items, greedy):
  0.103 at step 0, 0.619 at step 50, 0.675 at step 100, 0.528 at step 150,
  0.346 at step 200, 0.208 at step 250, 0.239 at step 300, 0.243 at step 350,
  0.251 at step 400. The final value is 42.4 points below the step-100 peak.
- In-distribution held-out accuracy (`split: val`, 500 items): 0.602 at step
  0, then 0.964, 0.966, 0.930, 0.912, 0.972, 0.978, 0.978 and 0.980 at step
  400.
- Parsed out-of-distribution predictions in the range 3-10: 206 at step 50,
  225 at step 100, 343 at step 150, 496 at step 200, 612 at step 250 and 626
  at step 400, out of 1,000. Only 136 gold answers are 3 or larger.
  Predictions in the range 0-2 fall from 787 at step 50 to 337 at step 400.
- Accuracy on the 864 items whose gold answer is 0-2: 0.654 at step 50, 0.734
  at step 100, 0.219 at step 250, 0.272 at step 400.
- Late outputs describe the whole scene and return its size. Example at step
  400: question "How many metal biplanes have the same size as the rubber
  truck?", gold 0, output lists eight vehicles and answers 8.
- `rewards/count_reward` averages 0.886 over steps 1-50 and stays between
  0.929 and 0.978 over every later 50-step block.
  `rewards/strict_format_reward` is 0.99 from step 51 on.

## Symptoms, not root causes

- The share of out-of-distribution outputs with a parsable answer dips from
  0.998 at step 100 to 0.898 at step 250 and recovers to 0.963 at step 400.
  Accuracy does not recover with it.
- `completion_length` grows from 35 tokens (steps 51-100) to 81 tokens (steps
  251-300) as the reasoning turns into a scene inventory.
- `kl` averages 0.82 over the first 50 steps and falls to 0.07 by the end.

## Not the root cause

- **The reward functions.** `count_reward` and `strict_format_reward` do what
  they say and in-distribution accuracy reaches 0.980. The reward is correct
  for the training questions; the training questions are the problem.
- **Learning rate (`2e-6`) and KL coefficient (`--beta 0.01`).** They set how
  fast the policy moves, not where it moves. Citing them as the root cause
  without the data is a wrong attribution. A diagnosis may mention them as
  amplifiers.
- **The step budget or missing early stopping.** Stopping at step 100 keeps a
  better checkpoint, but it selects on the out-of-distribution set and leaves
  the cause in place; accuracy on items with gold 3 or larger is already
  falling by step 100 (0.397 at step 50, 0.301 at step 100).
- **"Training did not learn" or "the run diverged".** The run learns its
  training distribution almost perfectly.
- **The evaluation script, the answer regular expression or the 256-token
  limit.** The same script scores step 0, the peak and the end, and at most
  10.2% of outputs are unparsed at any step after step 0.
- **A bug in the trainer or the prompt template.** None is involved.

## Evidence

A passing diagnosis reads the datasets, not only the logs. It should cite at
least: (1) the single question and the 3-10 answer range in `data/train`
against the question variety and the 0-2 answers of `data/ood_val`; (2) the
out-of-distribution accuracy series with its peak near step 100 and its fall,
against flat or rising in-distribution accuracy; (3) evidence from
`eval_outputs/` that late predictions move into the training answer range or
ignore the question. Exact counts are not required when the direction and
rough size are right.
