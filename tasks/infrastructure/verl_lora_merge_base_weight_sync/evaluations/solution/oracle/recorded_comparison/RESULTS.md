# Calibration results — verl_lora_merge_base_weight_sync solution grader

Fresh 60-step reruns (one GPU, image 25607123), graded live by
`verifier/tests/grade_submission.py` with the full stage environment.

## Arms

| arm | step-60 strict acc (grader re-score) | logged `val-core` | dump own acc | verdict |
| --- | --- | --- | --- | --- |
| defective (workspace as shipped) | 0.0781 | 0.0735 | 0.0735 | **FAIL** (gate 0.4) — as expected |
| repaired (fix.diff applied) | 0.7786 | 0.7763 | 0.7763 | **PASS** (gate 0.4) — as expected |

Repaired rerun (2026-11-07, devspace 1498750): val climbs 0.7278@5 → 0.7763@60
(frozen reference 0.7771@60), `rollout_actor_probs_pearson_corr` 0.999,
`rollout_corr/kl` ~0.001 — the merged policy is actually served.

Defective rerun (2026-11-07, devspace 1498749): 60 steps, all 13 validation
points pinned at the base model's level (0.0735-0.0849), grader re-score
0.0781 vs frozen reference re-grade 0.0887 — same band. Delivery probe
(2026-11-07, gpprobe2 on 1498753): the rollout worker receives the bit-exact
base weight at every sync (q_proj abs-sum 78189.4375) while the engine-side
dict reads merged values inside `merged_lora_context` — hence the v2 fix
(`.detach().clone()` inside the context; see oracle README). gpprobe3
confirmed both the mechanism (raw chain flips to base across the context
exit, clone retains merged) and the fix (receiver then gets the merged
tensor, abs 78540.58 = engine-side value, live vLLM qkv abs changed).

Frozen reference (task creation, 2026-10-25): defective step 0 / step 60 =
0.0857 / 0.0842; repaired step 60 = 0.7771. Local re-grade of the frozen
defective dump: 0.0887 (grader rule is slightly more lenient than verl's own
scorer, 0.0842 — hence the agreement check compares the logged metric with the
dump's own per-sample accuracy, not with the grader's re-score).

## Verdict files

`outputs/lora-calib-defective.json`, `outputs/lora-calib-repaired.json`.
