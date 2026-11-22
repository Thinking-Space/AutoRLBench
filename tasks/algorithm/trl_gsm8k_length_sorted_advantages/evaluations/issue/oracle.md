# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

- `workspace/trl/trainer/grpo_trainer.py::GRPOTrainer._sort_by_completion_length`
  - kind: `symbol`
  - observed: this method is a local addition to the vendored TRL 0.27.2
    tree. `_prepare_inputs` calls it on every generation batch, after TRL's
    `shuffle_sequence_dict` and before `split_tensor_dict` cuts the batch
    into the four gradient-accumulation micro-batches. It computes
    `order = torch.argsort(completion_mask.sum(1))` and applies `value[order]`
    only to tensors with `value.dim() >= 2`. Prompt and completion ids and
    masks, `old_per_token_logps`, `sampling_per_token_logps` and the `(B, 1)`
    sequence-level `importance_sampling_ratio` are reordered together.
    `advantages` has shape `(B,)`, so it is passed through in the pre-sort
    order. `compute_loss` multiplies the per-token log-probabilities of the
    completion in row `i` by `advantages[i]`, which is the advantage of the
    completion that was in row `i` before the sort, in general a different
    completion, often of a different prompt. Each completion is reinforced
    or suppressed according to another completion's result.
  - expected: `advantages` reordered with the rest of the batch (e.g.
    `value.dim() >= 1`, or indexing `advantages` explicitly), or no sort.

A passing diagnosis must name this reordering step as the cause, say that
`advantages` (one-dimensional) is not reordered while the token tensors are,
and connect it to GRPO: the advantage applied to a completion is no longer
its own, so the policy gradient carries no consistent signal toward correct
answers or toward the required `#### <number>` final line.

A diagnosis earns only partial root-cause credit, and cannot pass, if it
names only one of these:

- "advantages are misaligned / credit assignment is broken" without locating
  the length sort and the `dim() >= 2` filter;
- the length sort or the padding trim as "risky" without the
  `advantages`-shape mechanism;
- `_trim_completion_padding`. It cuts columns that are padding in every row
  of a micro-batch and is correct, and the `dr_grpo` normalizer uses
  `max_completion_length`, not the trimmed width.

## Observable impact

- Primary source evaluation metric: `eval/gsm8k_accuracy`. It uses greedy
  decoding on the GSM8K test set (1,319 items), and an output cut off at
  the 1,024-token limit counts as wrong.
  - `workspace/results/eval_before.json`: 0.0144 at step 0.
  - `workspace/results/eval_after.json`: **0.0000** at step 100.
  - The share of outputs with a `#### <number>` line goes from 0.021 to
    0.000, and the share cut off at the token limit from 0.004 to 0.280.
  - A correctly wired run of this workspace reaches 0.72-0.75.
- `workspace/results/metrics.jsonl` (100 steps, 4 prompts x 8 generations
  per step):
  - 20-step block means of the batch reward: 0.102, 0.036, 0.017, 0.044,
    0.028. The in-order control: 0.350, 0.736, 0.780, 0.788, 0.845.
  - `frac_reward_zero_std` 0.46 over steps 1-20, so more than half of the
    groups have mixed rewards and a usable advantage signal at the start;
    it rises to 0.75-0.86 afterwards.
  - `grad_norm` 0.002-0.005 throughout: the optimizer is stepping.
- `workspace/results/rollouts.jsonl` logs every fourth completion (800 rows)
  with its prompt text, the reward and the advantage. It is written before
  the sort. Re-checked against the gold answers (join the prompt to the
  GSM8K training question), all 32 correct rows carry reward 1 and no wrong
  row does. The reward path, including the local `reward_num_workers`
  thread pool, is correct.
- The process exits 0 and nothing is logged as an error or warning.

## Trigger conditions

- Any GRPO run of this vendored tree. The sort runs on every generation
  batch whatever the configuration; nothing in `train_grpo.py` enables it.
- How the policy drifts without a consistent signal varies between runs. In
  six 100-step runs of the as-shipped workspace, four end at 0.000-0.008
  test accuracy and two drift onto the format and reach 0.27 and 0.43,
  against 0.72-0.75 for three in-order controls. A diagnosis does not need
  to mention this, but it must not use it to argue that the code is
  correct.

## Symptoms, not root causes

- Batch reward decaying from 0.10 to about 0.03, and held-out accuracy going
  from 1.4% to 0%.
- `frac_reward_zero_std` near 0.8 later in the run. Once the policy stops
  producing correct `####` answers, most groups are all-zero.
- Rising truncation: 6% of training completions in steps 81-100 and 28% of
  held-out outputs at step 100 reach the 1,024-token limit.
- `sampling/importance_sampling_ratio/mean` between 0.11 and 0.16. This is
  TRL's default sequence-level vLLM importance-sampling correction with
  `top_p=0.95`; the in-order control shows 0.11-0.32 and reaches 0.74.

## Not the root cause

- **The reward function, `check_answer.py`, `common.extract`, the answer
  format, or `reward_num_workers=32` and its thread pool
  (`_call_reward_func_concurrently`).** The slices are concatenated in
  submission order, and every logged reward matches re-checked correctness.
  The control keeps all of this unchanged.
- **"The reward is too sparse" / "the task is too hard for the model".**
  The base model already writes the `####` line on 2% of test items, 54% of
  groups have mixed rewards over steps 1-20, and with aligned advantages the
  batch reward is 0.74 by steps 21-40.
- **Learning rate (`1e-5`), `beta=0.0`, temperature 1.0, LoRA rank, or the
  `dr_grpo` loss.** The control uses the same values.
- **vLLM importance-sampling settings, `top_p`, truncation, the 1,024-token
  limit, or the evaluation script.** All are the same in the control.
- **`_trim_completion_padding`.** It only removes all-padding columns.
- **"Training did not run" or "the optimizer is broken".** Gradients are
  non-zero throughout and the policy changes; it does not move toward the
  rewarded behaviour.

## Evidence

A passing diagnosis should cite at least the following. Exact counts are
not required when the direction and rough size are right.

1. The `dim() >= 2` filter in the vendored `_sort_by_completion_length`
   (called from `_prepare_inputs`), and that `advantages` has shape `(B,)`
   and is therefore not reordered with the completions it belongs to.
2. Evidence that a correct reward and advantage signal existed but was not
   followed: e.g. rollout rewards that match correctness, non-zero
   `reward_std` / `frac_reward_zero_std` well below 1 at the start, and the
   batch reward decaying instead of rising.
3. The held-out accuracy staying at about 0 (0.014 to 0.000).
