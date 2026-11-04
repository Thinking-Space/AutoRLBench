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

`score_mode` is `best`: the highest finite `eval/nq_test` over the run. That
value must be at least `0.28` and at most `1.0`. Pass is score `1`; all
failures are score `0`.

Four measured runs on devspace 1107990 (`/user/user2/harbor-runs/`), best
over steps `0..99`:

| run | code | best `0..99` |
| --- | --- | --- |
| `slime_search_r1__stage__env` 2026-10-10 | `loss_mask += [0]` (repaired) | **0.3733** |
| `slime_search_r1__stage__env` 2026-10-09 | repaired | **0.3570** |
| `slime_search_r1_nomask__stage__env` 2026-10-09 | `loss_mask += [1]` (defect) | **0.2237** |
| `slime_search_r1_nomask__stage__env` 2026-10-10 | defect (= the frozen evidence) | **0.1587** |

- Repaired margin above the bar: `0.077` (thinnest) and `0.093`.
- Defective margin below the bar: `0.056` (thinnest) and `0.121`.
- An untrained step 0 is about `0.045`–`0.051` and fails by a wide margin.
- The upper bound only blocks a fabricated value: accuracy cannot exceed `1.0`.

## Why the horizon is 99 and not 199

The defect is **high-variance**, and that variance grows with steps. The two
defective runs share identical code (`generate_with_search.py` unchanged since
its `f1177ea` commit), identical args apart from `--num-rollout`, a constant LR
schedule, and the same retrieval server — yet one collapsed to ~9-token
responses by step 20 and plateaued at `0.1657`, while the other kept ~650-token
responses and was still climbing at `0.2713` when it was killed at step 119.

At step 99 both defective runs are **fully observed**, so the gap
`[0.2237, 0.3570]` is measured on both sides. At step 199 the defective side is
**censored**: the higher of the two runs never finished, so no threshold there
can be shown to reject it. Do not move `final_step` to 199 without first
running a defective 200-step run to measure the ceiling.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.
