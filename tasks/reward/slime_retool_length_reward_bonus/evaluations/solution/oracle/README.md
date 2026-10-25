# Sealed implementation oracle

## Verdict

**unhealthy.** The staged ReTool recipe reproduces the canonical SFT→RL
tool-use setup on an unmodified upstream `slime` snapshot, but a length-reward
defect has been injected into `generate_with_retool.py::reward_func`.

## Root cause

`length_reward` — `examples/retool/generate_with_retool.py::reward_func` adds
`5 * min(sample.response_length / args.rollout_max_response_len, 1.0)` to
`result["score"]` for every sample, unconditionally, on top of the math_dapo
exact-match score. `rollout_max_response_len` is 16384. The bonus depends only
on generation length, so GRPO advantages reward longer rollouts and the policy
reward-hacks response length instead of accuracy.

## Required remediation

Remove the length-proportional term from `reward_func` so the reward depends
only on answer correctness (math_dapo exact-match with `strict_box_verify=True`)
plus the existing bounded tool-call shaping term. Do not replace it with any
other length- or verbosity-dependent bonus.

## Historical evidence

Evidence kind: `historical_run`. The frozen window is train step 100; the last
held-out eval is step 99 (`eval_interval=10`). Machine-readable copies:

- `oracle/historical_evidence_manifest.json`
- `oracle/dirty_reference/historical_metrics.json`
- Agent-visible `workspace/outputs/metrics.json`,
  `workspace/outputs/historical_eval_trace.log`, and the slimmed W&B
  `workspace/wandb/*/files/output.log`

Logged (polluted) `eval/aime` at step 99 is `+3.841`, up from `+0.822` at
step 0. Eval response length is about 15310 tokens with an 82.5% truncate
ratio. Train response length at step 100 is about 15159 tokens with a 78.5%
truncate ratio.

Because `eval/aime` is the mean of `reward_func`, the length term is inside
the logged metric. Subtracting `5 * min(mean_effective_len / 16384, 1)`
reconstructs decontaminated scores of about −0.417 at step 0 and −0.831 at
step 99. `reward_func` floors an incorrect sample at `min(-0.6, …)` rather than
`-1`, so those scores bound exact-match accuracy at no more than about 29.2%
and 8.4%; the true values are lower. Either way the held-out signal collapses
while the polluted metric rises.

## Why it is unhealthy

- `examples/retool/generate_with_retool.py::reward_func`: after
  `math_dapo_compute_score(..., strict_box_verify=True)` and the bounded
  tool-call shaping (`min(-0.6, …)` on already-failing samples), the function
  adds an unconditional `5 * min(response_length / rollout_max_response_len,
  1.0)` length bonus. This is the injected defect that drives reward hacking.

The surrounding recipe is otherwise correct and must not be reported as the
cause:

- `examples/retool/generate_with_retool.py`: multi-turn rollout injects
  code-interpreter `<interpreter>…</interpreter>` observations with
  `trainable=False` (`loss_mask=0`); only model-generated tokens are trained.
  With `return_logprob=True`, token ids and log probs come straight from
  `output_token_logprobs`; a missing-logprobs response aborts the sample instead
  of desyncing, per-turn `len(response_token_ids) == len(rollout_log_probs)` is
  asserted, and stale rollout state is reset on retried samples — so TIS stays
  aligned. Per-turn `max_new_tokens` is clamped to the remaining context budget
  and overflowing tool output is trimmed, so no sample exceeds the training
  token budget.
- `examples/retool/tool_sandbox.py`: the `code_interpreter` tool runs in a
  resource-limited subprocess sandbox (time/memory limits, cleanup), invoked
  under a concurrency semaphore.
- `examples/retool/retool_qwen3_4b_rl.sh`: standard GRPO
  (`--advantage-estimator grpo`, `--use-kl-loss --kl-loss-coef 0.00
  --kl-loss-type low_var_kl`, `--eps-clip 0.2 --eps-clip-high 0.28`,
  `--entropy-coef 0.00`), Adam `--lr 1e-6` constant schedule, and the recipe is
  wired only through the documented `--custom-generate-function-path` /
  `--custom-rm-path` hooks.

## Distractors that must not be reported as the defect

- `kl-loss-coef 0.00`, `max_turns=16`, `max_tool_calls=16`,
  `tool_concurrency=32`.
- `--rotary-base 5000000` for the Qwen3-4B long-context checkpoint.
- W&B `--use-wandb` and the `debug/*` payload logging; the `pkill`/`ray stop`
  rerun preamble.
- The optional SFT warm-up stage (`retool_qwen3_4b_sft.sh`) — skipping it and
  starting RL from the released SFT checkpoint is supported.
- Runtime asset paths under `/root/…` fulfilled on the training host.
- Rising logged `eval/aime` without subtracting the length bonus.

## Machine grading

Machine grading reads only the new experiment logs under
`$AUTORLBENCH_RESULTS/experiment_logs`. The required held-out metric is
`eval/aime`, read in the math_dapo **score** space `[-1, 1]` — the space the
frozen trace was logged in (`eval_reward_key: score`) and the space
`slime_retool` grades the same recipe in. No `--eval-reward-key` is involved;
the instruction fixes `--num-rollout 70` and `--eval-interval 10`, so the run
has eval points 0, 9, …, 69.

Grading the score channel is the discriminator, because that channel is where
the defect lives:

- the length term is still present → `eval/aime` leaves `[-1, 1]`
  (historically `+0.822` at step 0 up to `+3.841` at step 99)
- the length term is removed → `eval/aime` is the correctness signal, which
  cannot exceed `+1`

`score_mode` is `best` (highest finite point over the run; a single eval over
30 AIME problems is noisy). That value must be at least `-0.55` and at most
`0.5`. Pass is score `1`; otherwise `0`.

Both bounds are calibrated on the repaired reference run — the canonical recipe
with the length term removed, same image and assets, `--num-rollout 70`
(`devspace 1177503:/user/chenyinghao/harbor-runs/manual-slime-retool`):

| step | 0 | 9 | 19 | 29 | 39 | 49 | 59 | 69 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `eval/aime` | -0.5808 | -0.5706 | -0.5078 | -0.5035 | -0.5065 | -0.4373 | **-0.4246** | -0.4861 |

The lower bound fails an untrained step 0 (`-0.5808`) while the best point
(step 59, `-0.4246`) clears it by `0.125`. The upper bound sits below every
eval point an unrepaired run can produce — the thinnest case is step 0, where
the length bonus alone is `5 * 4042 / 16384 = +1.233` and the total lands near
`+0.65` — yet `0.92` above the repaired run's best point, so it cannot fail a
correct repair. `historical_final_baseline` in `grader_spec.json` is the
*defective* run's step-69 value (`+3.580`): here a repaired run must come in
below the baseline, not above it.

Do not quote an accuracy threshold for this level. `reward_func` floors an
incorrect sample at `min(-0.6, …)` rather than `-1`, so mean score does not
convert to accuracy by `(score + 1) / 2`; `-0.4861` only bounds the repaired
run's accuracy to `[7.1%, 25.7%]`, and no accuracy was ever measured for the
repaired recipe. The score channel is what was measured, so the score channel
is what is graded.
