# Grading-path evidence (reviewer reference, not used for grading)

## Runs and graders

- Four 100-step runs of `run_grpo.sh`: defective and repaired, seeds 1 and 2 (seed 1 on 2 GPUs per arm, seed 2 on
  1 GPU per arm). The two arms differ only in `verl/utils/reward_score/math_verify.py`: pre-fix revision
  `3d274f6` (defective) vs. the file as released at `6f2aa24` (repaired, a later rewrite of the #5635 fix that
  scores in a subprocess). The workspace freezes defective seed 1.
- Independent graders, both run in the main thread on the dumped generations:
  - `indep_mv`: math-verify 0.9.0 with latex2sympy2_extended 1.11.0 (antlr4-python3-runtime 4.9.3 from the image),
    `gold = parse("\boxed{" + gt + "}", (LatexExtractionConfig(),))`,
    `pred = parse(output, (ExprExtractionConfig(), LatexExtractionConfig()))`, correct iff `verify(gold, pred)`.
    No `math_metric` wrapper, called directly.
  - `indep_verl`: verl's `verl/utils/reward_score/math_reward.py::compute_score` at `6f2aa24` (last `\boxed{}` +
    string normalisation), an unrelated implementation.
- MATH-500 validation uses 497 of the 500 problems in every run. `filter_overlong_prompts=True` drops prompts whose
  chat-templated length (Qwen2.5-1.5B-Instruct tokenizer, `add_generation_prompt=True`) exceeds
  `data.max_prompt_length=512`: `test/counting_and_probability/181.json` (711 tokens),
  `test/counting_and_probability/282.json` (744), `test/prealgebra/1044.json` (835). No other problem is missing
  (`math500_excluded.txt`).

## 1. The exception that `except Exception: pass` swallows

`capture_stack.txt` runs exactly the body of the pre-fix `compute_score`'s `try` on one correct answer, once in the
main thread (score 1.0) and once through `loop.run_in_executor` as `NaiveRewardManager.run_single` does. In the
thread, math-verify's timeout wrapper calls `signal.signal(signal.SIGALRM, handler)` (`math_verify/utils.py`,
line 60), which raises `ValueError: signal only works in main thread of the main interpreter`; `parse()`
(`math_verify/parser.py`, line 711) re-raises it as `ValueError: Math-Verify 'parse' function doesn't support
threaded environment due to usage of signal.alarm() in timeout mechanism ...`. Python restricts installing signal
handlers (`signal.signal`) to the main thread; that is the call that fails.

## 2. The same answers through every grader

Step-100 MATH-500 generations (497 answers each), confusion against `indep_mv`:

| answers | correct / wrong | pre-fix, in thread | pre-fix, main thread | released, in thread | `indep_verl` |
| --- | --- | --- | --- | --- | --- |
| defective seed 1 | 262 / 235 | TP 0, TN 235, FN 262 | agrees on 497/497 | agrees on 497/497 | TP 246, TN 235, FN 16 |
| repaired seed 1 | 287 / 210 | TP 0, TN 210, FN 287 | agrees on 497/497 | agrees on 497/497 | TP 270, TN 208, FP 2, FN 17 |
| repaired seed 2 | 287 / 210 | TP 0, TN 210, FN 287 | agrees on 497/497 | agrees on 497/497 | TP 273, TN 210, FN 14 |

The released scorer in a thread rejects all 210–235 wrong answers, so it is not a grader that returns 1.

**Real reward paths on identical answers.** At step 0 both arms run the same model with greedy decoding; 228 of
the 497 MATH-500 generations are byte-identical between the two seed-1 runs (the rest differ through batch
nondeterminism). On those 228 answers (156 correct, 72 wrong by `indep_mv`), the score each run's own reward loop
logged in its dump: repaired 156 TP / 72 TN (100% agreement), defective 0 TP / 72 TN / 156 FN.

## 3. Learning gain on MATH-500 (fixed endpoint, step 100)

Per-problem correctness by `indep_mv` on the same 497 problems (`paired_seed*.txt`):

| seed | repaired: step 0 → 100 (within-arm gained/lost, p) | defective: step 0 → 100 (gained/lost, p) | step-100 gap (only-repaired / only-defective, p, 95% CI) |
| --- | --- | --- | --- |
| 1 | 0.5493 → 0.5775, +2.8 (51/37, p=0.165) | 0.5634 → 0.5272, −3.6 (30/48, p=0.054) | +5.0 (53/28, p=0.0073, [+1.6, +8.7]) |
| 2 | 0.5493 → 0.5775, +2.8 (43/29, p=0.125) | 0.5634 → 0.5694, +0.6 (42/39, p=0.824) | +0.8 (42/38, p=0.738, [−2.8, +4.2]) |

Tests: exact two-sided McNemar, i.e. a binomial test on the discordant pairs, `p = min(1, 2·P[X ≤ min(b, c)])` with
`X ~ Binomial(b + c, 1/2)`; the 95% interval is a paired percentile bootstrap over the 497 problems (10,000
resamples, RNG seed 0). The repaired arm's own gain is not significant in either seed and the between-arm gap does
not replicate. A stable acceptance threshold based on MATH learning gain has not been established, so this task is
issue-level and its claim is limited to the grading path.

## Files

- `capture_stack.txt`, `math500_excluded.txt`, `regrade_defective_seed1.jsonl` (logged vs. re-graded scores by
  step and source), `crossgrade_*.txt`, `paired_seed*.txt`, and `thread_check.txt` (pre-fix vs. released
  `compute_score`, main thread vs. `run_in_executor`). Each cross-grading and thread-check output prints the
  sha256 of the `math_verify.py` that was actually imported, so the two implementations can be told apart.
- `released_math_verify.py`: `verl/utils/reward_score/math_verify.py` as released at `6f2aa24` (scoring in a
  subprocess pool), the implementation the repaired runs used. The workspace carries only the pre-fix file.
- `scripts/`: the scripts that produced the outputs. All outputs were regenerated from the packaged files with
  task-relative paths; from the task root after `prepare.sh`, for example:

  ```bash
  python3 evaluations/issue/reference/scripts/crossgrade.py \
      --dump workspace/outputs/val_generations/100.jsonl --buggy-verl workspace/verl \
      --released evaluations/issue/reference/released_math_verify.py --math-verify workspace/third_party/python
  PYTHONPATH=workspace/third_party/python python3 evaluations/issue/reference/scripts/regrade.py \
      --verl workspace/verl workspace/outputs/rollouts workspace/outputs/val_generations
  PYTHONPATH=workspace/third_party/python python3 evaluations/issue/reference/scripts/paired.py \
      evaluations/issue/reference/runs/repaired-seed1/val_generations workspace/outputs/val_generations 100
  ```

  `crossgrade.py` runs each verl scorer in its own process with only its own verl tree on `sys.path`; the
  released scorer runs from a temporary copy of `workspace/verl` with `released_math_verify.py` in place.
- `runs/`: sanitized console histories and step-0/100 validation generations of the other three runs (repaired
  data seeds 1 and 2, defective data seed 2).
- `reference.patch`: upstream fix #5635 (`3d274f6` → `9586c974`) for this file.
