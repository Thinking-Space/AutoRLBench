# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs `val/em` from
`$AUTORLBENCH_RESULTS/experiment_logs`, which is where `run.py --runs-dir` puts
`train.log` and the offline W&B directory.

`val/em` is exact-match accuracy in `[0, 1]` on the 128 held-out HotpotQA
questions, the same space the frozen run logged it in. The defect distorts the
policy's inputs, not the metric, so there is no polluted channel to correct for
and grading is a straight accuracy comparison.

`grade_submission.py` is byte-identical to
`tasks/slime_search_r1_nomask/evaluations/solution/verifier/tests/`. Its step-line
pattern already fits this trainer's output:

```text
step:32 - val/test_score/hotpotqa:0.156 - val/em:0.156 - val/f1:0.247 - ...
```

One consequence of reusing it unchanged: its step-line pattern is line-anchored
(no `re.DOTALL`), so a match never spills across lines and `best_step` is the
step the best eval actually happened at. Replayed on the two frozen logs it
reports the right values (`0.156` defective, `0.328` repaired) with
`best_step: 32`.

## Horizon

`config.json` fixes the budget at `steps` 32 / `eval_every` 10, and the
instruction forbids changing it, so the rerun has eval points 0, 10, 20, 30, 32
and `final_step` is `32` — the same horizon the frozen evidence covers. There is
no censored-tail problem here: both recorded runs completed all 32 steps.

## Bounds and how they are calibrated

`score_mode` is `best`: the highest finite `val/em` over the run. That value must
be at least `0.24` and at most `1.0`. Pass is score `1`; all failures are `0`.

The recorded pair (`oracle/recorded_comparison/comparison.json`), same seed,
data, encoder, corpus, index and budget, differing only in the four defective
lines:

| run | code | step 0 | best over 0..32 |
| --- | --- | ---: | ---: |
| recorded defective (= the frozen evidence) | unwritten `response_buffer` | 0.03125 | **0.15625** |
| recorded control | `.detach().cpu().numpy()` | 0.13281 | **0.32813** |

- Repaired margin above the bar: `0.088`.
- Defective margin below the bar: `0.084`.
- A correct fix with no training scores about `0.133` at step 0 and fails, so the
  bar requires repairing *and* completing the run.
- The upper bound only blocks a fabricated value: accuracy cannot exceed `1.0`.

## Calibration spread

**Two runs per side.** A second pair was measured on 2026-11-03 (devspace
reruns, both exit 0 at the same budget): defective best `0.156` (steps
0/10/20/30/32 = 0.023/0.031/0.094/0.156/0.141) and repaired best `0.344`
(0.133/0.219/0.305/0.344/0.328, final 0.328 matching the recorded 0.32813).
Both defective runs cap at `0.156`, so the bar keeps 0.084 headroom above the
defective ceiling on both pairs and 0.084-0.104 below the repaired values.

`slime_search_r1_nomask` still has the wider four-run calibration; two pairs
here agree on the separation but do not establish seed-to-seed spread, and the
defective side in particular is the kind of defect that can wander: the policy
still gets reward from its parametric knowledge, so its ceiling depends on how
fast it adapts to useless observations.

Consequences for whoever reads a result:

- a best value near `0.24` remains inside a thinly measured band — read
  `experiment_logs/graded/train.log` and `retrieval.jsonl` before trusting the
  verdict either way,
- reruns are not expected to be bit-identical in any case: the FAISS index is
  rebuilt at prepare time rather than shipped, and GPU nondeterminism applies.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.
