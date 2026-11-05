# Sealed implementation oracle

## Verdict

**unhealthy.** The staged ReTool recipe reproduces the canonical SFT→RL
tool-use setup on an unmodified upstream `slime` snapshot, but two defects
have been injected into `examples/retool/generate_with_retool.py`: a
length-reward defect in `reward_func` and a tool-call argument-key mismatch
in the tool-call parser.

## Root cause

Two independent causes; a complete attribution names both.

- `length_reward` — `reward_func` adds
  `5 * min(sample.response_length / args.rollout_max_response_len, 1.0)` to
  `result["score"]` for every sample, unconditionally, on top of the math_dapo
  exact-match score. `rollout_max_response_len` is 16384. The bonus depends
  only on generation length, so a wrong-but-long rollout can outscore a
  correct-but-short one; under GRPO the resulting advantages reward longer
  generations and the policy reward-hacks response length instead of
  accuracy.
- `tool_call_argument_key_mismatch` — the chat template (`TOOL_TEMPLATE`)
  instructs the model to emit tool calls as
  `{"name": <function-name>, "arguments": <args-json-object>}`, but the
  parser reads `tool_call_data.get("parameters", {})`. The key mismatch
  means `arguments` is always the empty dict, the extracted `code` is `""`,
  and the interpreter branch is skipped; only the legacy `<code>…</code>`
  fallback can still execute anything. The tool channel is therefore broken
  for well-formed calls — which is why even the step-0 eval (before any RL)
  already sits at a decontaminated −0.84.

## Required remediation

Remove the length-proportional term from `reward_func` so the reward depends
only on answer correctness (math_dapo exact-match with
`strict_box_verify=True`) plus the existing bounded tool-call shaping term;
do not replace it with any other length- or verbosity-dependent bonus. And
parse the arguments under the key the template defines:
`tool_call_data.get("arguments", {})`. Fixing only one of the two leaves the
run outside the passing band (measured single-fix state: `−0.845` at step 69
with only the length bonus removed).

## Historical evidence

Evidence kind: `historical_run`. The frozen window is train step 70; the last
held-out eval is step 69 (`eval_interval=10`). Machine-readable copies:

- `oracle/historical_evidence_manifest.json`
- `oracle/dirty_reference/historical_metrics.json`
- Agent-visible `workspace/outputs/metrics.json`,
  `workspace/outputs/historical_eval_trace.log`, and the slimmed W&B
  `workspace/wandb/*/files/output.log`

Logged (polluted) `eval/aime` at step 69 is `+3.0138`, up from `+0.3556` at
step 0. Eval response length is about 12394 tokens with a 47.3% truncate
ratio. Train response length at step 69 is about 9250 tokens with a 13.7%
truncate ratio.

Because `eval/aime` is the mean of `reward_func`, the length term is inside
the logged metric. Subtracting `5 * min(mean_effective_len / 16384, 1)`
reconstructs decontaminated scores of about `−0.839` at step 0 and `−0.769`
at step 69. `reward_func` floors an incorrect sample at `min(-0.6, …)`
rather than `-1`, so those scores bound exact-match accuracy at no more than
about 8.0% and 11.6%; the true values are lower. The held-out signal is
broken across the whole window — already at step 0, because the
argument-key mismatch breaks the tool channel before RL starts — while the
polluted metric rises.

## Why it is unhealthy

- `examples/retool/generate_with_retool.py::reward_func`: after
  `math_dapo_compute_score(..., strict_box_verify=True)` and the bounded
  tool-call shaping (`min(-0.6, …)` on already-failing samples), the function
  adds an unconditional `5 * min(response_length / rollout_max_response_len,
  1.0)` length bonus. This is one of the two injected defects; it drives the
  reward hacking.
- `examples/retool/generate_with_retool.py` (the tool-call parser): the
  template/`parameters` key mismatch means correctly formatted calls extract
  no `code`, so the interpreter never executes them. This is the other
  injected defect; it is why held-out correctness is already at the floor
  before RL begins.

The surrounding recipe is otherwise correct and must not be reported as the
cause:

- `examples/retool/generate_with_retool.py`: multi-turn rollout injects
  code-interpreter `<interpreter>…</interpreter>` observations with
  `trainable=False` (`loss_mask=0`); only model-generated tokens are trained.
  With `return_logprob=True`, token ids and log probs come straight from
  `output_token_logprobs`; a missing-logprobs response aborts the sample
  instead of desyncing, per-turn `len(response_token_ids) == len(rollout_log_probs)` is
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

`score_mode` is `final`: the graded value is the `eval/aime` at step 69, the
run's final eval point, and a run that never reaches step 69 fails outright.
That value must land in `[-0.55, 0.5]`. Pass is score `1`; otherwise `0`.

Grading the score channel is the discriminator, because the reward defect
lives in it: with the length bonus still present `eval/aime` leaves `[-1, 1]`
and trips the upper bound; with the term removed it is the correctness
signal, which cannot exceed `+1` — but only if the tool channel is also
repaired, since the parser mismatch alone holds it at about `−0.845`.

Calibrated on measured runs at the graded horizon (step 69; devspace 1177503
`/user/user2/harbor-runs/manual-slime-retool` for the repaired
reference):

| run | code | value at 69 |
| --- | --- | --- |
| `manual-slime-retool` 2026-10-30 | both defects fixed | **-0.4861** |
| `calib2-retool-bonly` 2026-11-05 | length bonus removed, arguments key left broken | -0.845 |
| `calib2-retool-ab` 2026-11-04 | both defects (= the frozen evidence) | +3.0138 |

The repaired run's best point is step 59 at `-0.4246` and its untrained step
0 is `-0.5808`. Thinnest margin is `0.0639` on the passing side
(`-0.4861 - (-0.55)`); the closest failing measurement is the length-only
repair, `0.295` below the lower bound; the dual-defect evidence exceeds the
upper bound by `2.51`.

Do not quote an accuracy threshold for this level. `reward_func` floors an
incorrect sample at `min(-0.6, …)` rather than `-1`, so mean score does not
convert to accuracy by `(score + 1) / 2`; `-0.4861` only bounds the repaired
run's accuracy to `[7.1%, 25.7%]`, and no accuracy was ever measured for the
repaired recipe. The score channel is what was measured, so the score channel
is what is graded.
