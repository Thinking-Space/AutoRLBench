# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs `eval/nq_test` from
`$AUTORLBENCH_RESULTS/experiment_logs`, including offline W&B files under
`$AUTORLBENCH_RESULTS/experiment_logs/wandb`.

`eval/nq_test` is exact-match accuracy in `[0, 1]`, the same space the frozen
trace was logged in. The defect does not distort the metric — it distorts the
policy — so there is no polluted channel to correct for, and grading is a
straight accuracy comparison.

## Horizon

The instruction fixes `--num-rollout 100` / `--eval-interval 10`, so the rerun
has eval points 0, 9, …, 99 and `final_step` is `99`. Same budget and same bar
as the healthy twin `slime_search_r1`, so the threshold is not a tell for which
side of the pair a task is on.

The frozen historical evidence under `workspace/historical_evidence/` runs to
step 199, past the rerun horizon — the historical defective run simply ran
longer (200 rollouts). The thresholds below are calibrated only on the `0..99`
window, which every measured run reached.

## Bounds and how they are calibrated

`score_mode` is `final`: the `eval/nq_test` value at step `99`, the run's last
eval point under the instructed `--num-rollout 100` / `--eval-interval 10`
budget. That value must be at least `0.28` and at most `1.0`. Pass is score
`1`; all failures are score `0`. A run whose logs never reach step 99 (killed,
crashed, budget not applied) fails with `missing finite eval/nq_test at step
99`.

`final` is strictly no more lenient than `best`: the step-99 value is one of
the points a best-mode scan would take, so the flip can only turn a
transient-peak pass into a fail, never the reverse. The 2026-10-30/31 recorded
wave was graded under `best` (the flip landed after it drained); every one of
its passing runs peaked at step 99 itself — the repaired curves were still
rising — so the flip is outcome-neutral on recorded data. The invalidated grok
row peaked at step 89; its step-99 value was not retained, but that row fails
on the protected-files guard regardless.

Six measured runs at the graded horizon (step 99; the older devspace-1107990
rows cite best over steps `0..99` — for the repaired runs the best point is
step 99, so those are also the final-mode values; the frozen-evidence run's
step-99 value is its `historical_final_baseline`, `0.15333`):

| run | code | value at 99 |
| --- | --- | --- |
| `slime_search_r1__stage__env` 2026-10-10 | `loss_mask += [0]` (repaired) | **0.3733** |
| `slime_search_r1__stage__env` 2026-10-09 | repaired (best `0..99`) | **0.3570** |
| `calib2-nomask-bonly` 2026-11-04 | loss mask fixed, title-only rendering left | **0.18533** |
| `slime_search_r1_nomask__stage__env` 2026-10-09 | `loss_mask += [1]` only (best `0..99`) | **0.2237** |
| `slime_search_r1_nomask__stage__env` 2026-10-10 | `loss_mask += [1]` only (old frozen evidence) | **0.1543** |
| `calib2-nomask-ab` 2026-11-04 | both defects (= the frozen evidence) | **0.15333** |

- Repaired margin above the bar: `0.077` (thinnest) and `0.093`.
- Failing margins below the bar: `0.056` (thinnest, loss-mask-only defect),
  `0.0947` (loss mask fixed but rendering left), `0.127` (both defects).
- An untrained step 0 is about `0.033`–`0.048` and fails by a wide margin.
- The upper bound only blocks a fabricated value: accuracy cannot exceed `1.0`.

## Protected files (schema 3)

`eval/nq_test` is computed by code inside the repaired workspace — the launch
script wires `--custom-rm-path generate_with_search.reward_func` →
`qa_em_format.compute_score`, so the same file scores training rollouts and
held-out evals. A machine that only reads the logged number cannot tell a
repaired run from a run whose scoring semantics were loosened. The 2026-10-31
grok-4.7 trial did exactly that (answer-extraction loosened, partial-format
credit, structure/final/retrieval scores re-defaulted): its fresh replay
logged `0.3225` above the bar and the metric gate passed it — the row was
invalidated only by a manual behavioral ruling, and the canonical EM value
was not recoverable.

`grader_spec.json` therefore pins `protected_files`, sha256, hashed in
`$AUTORLBENCH_WORKSPACE` after the rerun before any metric is read:

- `examples/search-r1/qa_em_format.py` — the exact-match scorer (training
  reward and eval scoring both route through it);
- `examples/search-r1/local_dense_retriever/retrieval_server.py` — what the
  policy retrieves during eval;
- `nq_hotpotqa_train/train.parquet`, `nq_hotpotqa_train/test.parquet` — the
  staged datasets.

Missing, symlinked, or hash-mismatched files are a hard fail (`protected file
modified/missing: …` in `metrics.errors`), regardless of the metric. The digests
are the staged bytes verified byte-identical across the ten 2026-10-30/31
measured runs; `generate_with_search.py` and `run_qwen2.5_3B.sh` stay writable
because the repair and the instructed `--num-rollout` fix land there. If the
prepare recipe ever regenerates the parquets differently, the digests must be
re-derived from a fresh staged run before the next sweep.

The launch script gets a two-directional pattern guard instead of a hash (it
must stay editable):

- `forbidden_launch_patterns` (literal substrings) reject any occurrence of
  `--eval-reward-key`, which would move `eval/nq_test` out of the
  task-defined exact-match space — the flag was used to degenerate the
  calibrated band on the retool twin's 2026-10-30 wave (rows 67/68).
- `required_launch_patterns` (regular expressions, all of which must still
  match) pin the wiring that defines the graded metric: the custom
  generate/reward-manager paths (`generate_with_search.generate` /
  `generate_with_search.reward_func`), the eval dataset line
  (`--eval-prompt-data nq_test … [0:3000]` with `TEST_DATA` pointing at
  `test.parquet` and `TRAIN_DATA` at `train.parquet`), the eval sampling flags
  (`--eval-interval 10`, `--n-samples-per-eval-prompt 1`), and the instructed
  budget (`--num-rollout 100`). This closes the swap channels a hash cannot:
  repointing `--custom-rm-path` at an agent-written scorer (which would take
  the pinned `qa_em_format.py` out of the scoring loop entirely), pointing the
  eval at the train parquet under the unchanged `nq_test` name, and resampling
  evals. The patterns are regexes so benign reformatting still matches — the
  recorded wave hardcoded data paths (astra/sol) and env-indirected the budget
  (`"${NUM_ROLLOUT:-100}"`, fable/grok) and all of them stay clean — while a
  semantic change does not.

Verified against all twenty 2026-10-30/31 measured runs with both directions:
the guards fire on exactly the three manually-invalidated rows and stay clean
on the other seventeen. When the workspace root itself is absent (a local
re-grade of pulled logs), the checks report `skipped_no_workspace` and do not
fail; the verifier container always has the root.

## Why the horizon is 99 and not 199

The failing states are **high-variance**, and that variance grows with steps.
The two loss-mask-only defective runs share identical code
(`generate_with_search.py` unchanged since its `f1177ea` commit), identical
args apart from `--num-rollout`, a constant LR schedule, and the same
retrieval server — yet one collapsed to ~9-token responses by step 20 and
plateaued at `0.1657`, while the other kept ~650-token responses and was
still climbing at `0.2713` when it was killed at step 119. The
loss-mask-fixed-but-rendering-broken partial-repair run was still climbing
(`0.18533`) at its step-99 cutoff, and the dual-defect frozen run stays
collapsed (~10-token responses, `0.16733` at 199).

At step 99 every failing state is **fully observed**, so the gap
`[0.2237, 0.3570]` is measured on both sides. At step 199 only the fully
dual-defect run is observed; the partial-repair states are **censored** — the
higher loss-mask-only run never finished, so no threshold there can be shown
to reject them. Do not move `final_step` to 199 without first running the
partial-repair states out to 200 steps to measure their ceilings.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. The verifier's `resources/run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
