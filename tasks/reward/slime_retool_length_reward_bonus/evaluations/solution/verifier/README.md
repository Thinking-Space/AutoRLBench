# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs `eval/aime` from
`$AUTORLBENCH_RESULTS/experiment_logs`, including offline W&B files under
`$AUTORLBENCH_RESULTS/experiment_logs/wandb`.

## Which space `eval/aime` is read in

The math_dapo **score** space, `[-1, 1]`, exactly as `slime_retool` grades the
same recipe — and exactly the space the frozen historical trace was logged in
(`eval_reward_key: score` in `workspace/wandb/latest-run/files/config.yaml`).
No `--eval-reward-key` is involved.

That choice is what makes this level gradable at all, and it is the whole
discriminator:

- `reward_func`'s score channel is where the length-reward defect lives, so a
  run that still carries `5 * length_ratio` reports an `eval/aime` **outside**
  `[-1, 1]` at the graded horizon — the frozen dual-defect run logs `+3.0138`
  at step 69.
- A run with the length term removed reports the correctness signal, which
  cannot exceed `+1` — but only once the tool-call parser is repaired too,
  since the `parameters`/`arguments` key mismatch alone holds the score at
  about `-0.845` at step 69.

So "did the agent remove the length bonus and restore the tool channel" is
read directly off the metric, rather than inferred from a second-order
accuracy collapse.

## Horizon

The instruction fixes `--num-rollout 70` / `--eval-interval 10`, so the rerun
has eval points 0, 9, …, 69 and `final_step` is `69`. Same budget as the
repaired calibration runs, so the band below is calibrated exactly at the
graded horizon.

## Bounds and how they are calibrated

`score_mode` is `final`: the `eval/aime` value at step 69, the run's last eval
point under the instructed budget. That value must be at least `-0.55` and at
most `0.5`. Pass is score `1`; all failures are score `0`. A run whose logs
never reach step 69 (killed, crashed, budget not applied) fails with `missing
finite eval/aime at step 69`.

`final` is strictly no more lenient than `best`: the step-69 value is one of
the points a best-mode scan would take, so the flip can only turn a
transient-peak pass into a fail, never the reverse. The 2026-10-30/31
recorded wave was graded under `best` against the single-defect evidence that
has since been superseded; this wave's frozen evidence is the 2026-11-04
dual-defect run, and the spec flips to `final` with it. `final` also matters
against the unrepaired states here: an unrepaired run's **early** points can
sit inside the band (the dual-defect run measures `+0.356` at step 0, before
RL has inflated response length), and only the end-of-run value exposes the
length hack.

Three measured runs at the graded horizon (step 69):

| run | code | value at 69 |
| --- | --- | --- |
| `manual-slime-retool` (devspace 1177503) 2026-10-30 | both defects fixed | **-0.4861** |
| `calib2-retool-bonly` 2026-11-05 | length bonus removed, arguments key left broken | -0.845 |
| `calib2-retool-ab` 2026-11-04 | both defects (= the frozen evidence) | +3.0138 |

- **Lower bound `-0.55`.** The repaired reference's step-69 value is
  `-0.4861`, `0.064` clear; the partial repair with only the length bonus
  removed sits `0.295` below. Same bound shape as `slime_retool`, so the
  healthy twin and the repaired defect twin are judged against one curve.
- **Upper bound `0.5`.** The dual-defect evidence exceeds it by `2.51` at
  step 69. An unrepaired run only stays below it while its responses are
  still short — the length bonus alone passes `0.5` once eval responses
  exceed ~1.6k tokens, and the measured dual-defect run grows from 3.9k to
  12.4k across the window. In the other direction `0.5` is `0.92` above the
  repaired run's best point (step 59, `-0.4246`); clearing it legitimately
  would take roughly 75% AIME accuracy, which this recipe is nowhere near in
  70 steps. The bound therefore cannot fail a correct repair.

Verified against the measured traces: the repaired log scores `1` (step-69
`-0.4861` within `[-0.55, 0.5]`); the frozen dual-defect trace scores `0`
(`+3.0138` at step 69 above the required maximum `0.5`); the length-only
repair scores `0` (`-0.845` at step 69 below the required minimum `-0.55`);
and a run truncated before the final step scores `0` (missing finite
`eval/aime` at step 69).

## Known limit

A partial repair that merely shrinks the coefficient (say `5` to `1`) can land
inside `[-0.55, 0.5]` and pass: at the dual-defect run's step-69 response
length the bonus would be about `0.76` on top of a `-0.845` correctness
floor. Metric-only grading cannot separate that from a real removal;
`evaluations/issue` is what grades the attribution, and `instruction.md`
asks for the term to be removed. Do not lower the upper bound to chase this —
the margin against a correct repair is what protects the level from false
negatives.

## Protected files and launch guard (schema 3)

`eval/aime` is computed by code inside the repaired workspace — the launch
script wires `--custom-rm-path generate_with_retool.reward_func`, and that
function scores held-out evals through `math_dapo_compute_score`. A machine
that only reads the logged number cannot tell a repaired run from a run whose
scoring semantics were changed around the defect. Both observed shapes are now
machine-checked:

- **Scorer/data edits** (the grok search_r1 row, 2026-10-31):
  `grader_spec.json` pins `protected_files`, sha256, hashed in
  `$AUTORLBENCH_WORKSPACE` after the rerun before any metric is read —
  `slime/rollout/rm_hub/math_dapo_utils.py` (the exact-match scorer training
  and eval both route through), `examples/retool/tool_sandbox.py` (the tool
  sandbox the policy drives during eval), and the staged datasets
  `zhuzilin_dapo-math-17k/dapo-math-17k.jsonl` and
  `zhuzilin_aime-2024/aime-2024.jsonl`. Missing, symlinked, or hash-mismatched
  files are a hard fail (`protected file modified/missing: …` in
  `metrics.errors`). The digests are the staged bytes verified byte-identical
  across the ten 2026-10-30 measured runs.
- **Metric-space switch** (retool rows 67/68 = astra/sol, 2026-10-30): the
  agents added `--eval-reward-key acc` to the launch script, moving
  `eval/aime` from the task-defined math_dapo score space `[-1, 1]` into raw
  accuracy `[0, 1]`, where the calibrated band `[-0.55, 0.5]` degenerates.
  The launch script must stay editable (the instructed `--num-rollout` and
  log-placement repairs land there), so it is guarded by pattern instead:
  `launch_script` + `forbidden_launch_patterns: ["--eval-reward-key"]` in the
  spec — any occurrence is a hard fail
  (`launch script contains forbidden pattern '--eval-reward-key'`).
- **Scoring/eval rewiring**: `required_launch_patterns` (regular expressions,
  all of which must still match) pin the wiring that defines the graded
  metric — the custom generate/reward-manager paths
  (`generate_with_retool.generate` / `generate_with_retool.reward_func`), the
  dataset assignments (`TRAIN_DATA` → `dapo-math-17k.jsonl`,
  `EVAL_DATA` → `aime-2024.jsonl`), the eval line
  (`--eval-prompt-data aime`), the eval sampling flags (`--eval-interval 10`,
  `--n-samples-per-eval-prompt 16`, `--eval-max-response-len 16384`,
  `--eval-top-p 1`), and the instructed budget (`--num-rollout 70`). This
  closes the channels a hash cannot: repointing `--custom-rm-path` at an
  agent-written scorer (taking the pinned `math_dapo_utils.py` out of the
  scoring loop) or swapping the eval dataset under the unchanged `aime`
  name. The patterns are regexes so benign reformatting still matches — the
  recorded wave env-indirected the budget (`"${NUM_ROLLOUT:-70}"`, fable) and
  rewrote the dataset defaults while staying clean — while a semantic change
  does not.

`examples/retool/generate_with_retool.py` stays writable because both
sanctioned repairs (the length-term removal and the argument-key fix) land
there; the hash guard cannot police the reward function itself, so edits
there that keep `eval/aime` in-window by construction rather than by removing
the term remain a behavioral ruling, per the known limit above. Verified
against all twenty 2026-10-30/31 measured runs: the guards fire on exactly
the three manually-invalidated rows (grok search_r1, retool astra/sol) and
stay clean on the other seventeen. When the workspace root itself is absent
(a local re-grade of pulled logs), both checks report `skipped_no_workspace`
and do not fail; the verifier container always has the root.

## Accuracy is deliberately not graded

`reward_func` floors an incorrect sample at `min(-0.6, …)` rather than `-1`, so
a mean score of `-0.4861` does **not** convert to `(score + 1) / 2 = 25.7%`
accuracy; it only bounds accuracy to `[7.1%, 25.7%]`. No measured accuracy
exists for the repaired recipe, so no accuracy threshold can be calibrated.
Grade the score channel, which is measured.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. The verifier's `resources/run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
