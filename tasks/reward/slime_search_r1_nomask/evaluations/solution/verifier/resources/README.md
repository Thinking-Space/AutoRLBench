`test.sh` is the single entry point. It uses deterministic metric and behavioral checks; no LLM judgment. It strips ANSI escapes and reconstructs `eval/nq_test` from
`$AUTORLBENCH_RESULTS/experiment_logs`.

- `metric_key`: `eval/nq_test`, exact-match accuracy in `[0, 1]`
- `score_mode`: `final` — the value at step `final_step`
- `required_final_score`: `0.28` (lower bound; same bar as the healthy twin)
- `required_final_score_max`: `1.0` (accuracy cannot exceed 1; blocks a
  fabricated value)
- `final_step`: `99` — the last eval point of the instruction's
  `--num-rollout 100` budget, and the graded point: a run whose logs never
  reach step 99 fails with `missing finite eval/nq_test at step 99`.
- `historical_final_baseline`: `0.15333333333333332` — the frozen
  defective run's step-99 value.

Pass is score `1`; all failures are score `0`.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. The task-root `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.

## Functional acceptance (2026-11-23)

Passing requires the metric gate AND `tests/check_behavior.py` on the submitted
workspace. The probe executes the actual generator with mocked model/retriever
IO and an independent minimal Sample implementation: passage bodies must reach the next model
call, external observation tokens must have zero final loss weight, model tokens
must remain trainable, and response/tokens/masks must stay aligned across turns.
Equivalent implementations are accepted; no exact replacement string is required.

The same reward wrapper scores evaluation. Wrong answers, including correctly
formatted answers and wrong answers following a successful retrieval, must still
score zero; correct answers score one. Pinning qa_em_format.py alone did not
protect this contract: changing its caller's structure_format_score or
retrieval_score arguments changed eval/nq_test to a shaped reward. Such a number
cannot be compared to the binary-EM 0.28 threshold. The threshold and 100-rollout
budget remain unchanged; raising the threshold would not fix the semantic leak.

A log-only invocation now reports unavailable behavior evidence and cannot pass.
The behavior probe supplies its own minimal `Sample` implementation, so editing
the submitted framework's `Sample` class cannot make the probe accept an invalid
generator. It also checks append-level trainability flags and zero dummy
log-probabilities on retrieval observations. If two files report different
values for the same evaluation step, grading fails closed instead of selecting
the larger value.
Historical rejudgments may combine a preserved verified replay metric with
reconstructed final source and explicitly recorded behavioral checks; they do
not claim a new GPU replay. If scoring semantics changed, the historical scalar
is preserved as a shaped reward, not reinterpreted as exact-match accuracy.
