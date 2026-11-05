`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs `val/em` from
`$AUTORLBENCH_RESULTS/experiment_logs`.

- `metric_key`: `val/em`, exact-match accuracy on the 128 held-out HotpotQA
  questions, in `[0, 1]`. The trainer prints it on its own step lines
  (`step:32 - val/test_score/hotpotqa:0.156 - val/em:0.156 - ...`), which is the
  shape `grade_submission.py` already parses; `val/test_score/hotpotqa` is the
  same number under the dataset's name.
- `score_mode`: `best` — the highest finite point over the run
- `required_final_score`: `0.24` (lower bound)
- `required_final_score_max`: `1.0` (accuracy cannot exceed 1; blocks a
  fabricated value)
- `final_step`: `32` — the last eval of `config.json`'s fixed budget
  (`steps` 32, `eval_every` 10 ⇒ eval points 0, 10, 20, 30, 32). Informational
  under `score_mode: best`.
- `historical_final_baseline`: `0.15625` — the frozen defective run's step-32
  value, which is also its best point.

Pass is score `1`; all failures are score `0`.

## Why 0.24

Measured, one run per side, same seed / data / corpus / index / budget
(`evaluations/solution/oracle/recorded_comparison/comparison.json`):

| step | defective `val/em` | repaired `val/em` |
| ---: | ---: | ---: |
| 0 | 0.03125 | 0.13281 |
| 10 | 0.04688 | 0.23438 |
| 20 | 0.11719 | 0.31250 |
| 30 | 0.13281 | 0.32031 |
| 32 | 0.15625 | 0.32813 |

`best` over the run is `0.15625` defective and `0.32813` repaired, so `0.24`
sits 0.084 above the defective ceiling and 0.088 below the repaired value. It is
also above the repaired run's step-0 point (`0.13281`), so a correct fix with no
training does not pass either: the agent has to repair *and* complete the run.

**Two runs per side.** A second pair was measured on 2026-11-03 (devspace
reruns, both exit 0 at the same budget): defective best `0.156` (steps 0/10/20/30/32
= 0.023/0.031/0.094/0.156/0.141) and repaired best `0.344`
(0.133/0.219/0.305/0.344/0.328, final 0.328 matching the recorded 0.32813). Both
defective runs cap at `0.156`, so `0.24` keeps 0.084 headroom above the defective
ceiling on both pairs; the sibling task `slime_search_r1_nomask` still has the
wider four-run calibration, but the two pairs here agree on the separation.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. The task-root `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
