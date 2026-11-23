# Solution verifier: functional repair and held-out accuracy

`test.sh` is the single entry point. It uses deterministic metric and behavioral checks; no LLM judgment. It strips ANSI escapes and reconstructs `eval/nq_test` from
`$AUTORLBENCH_RESULTS/experiment_logs`, including offline W&B files under
`$AUTORLBENCH_RESULTS/experiment_logs/wandb`.

`eval/nq_test` must remain exact-match accuracy in `[0, 1]`, the same space the
frozen trace was logged in. The staged defects affect the policy, but a submitted
repair can also change the metric by changing the reward wrapper's arguments.
The functional check must establish binary-EM semantics before the logged value
can certify that the accuracy threshold was reached.

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

### 2026-11-23：两条假通过与有效修复的对比

以下均为现有提交的历史 step-99 指标；没有为本次对比重新训练。
实现结论来自最终提交代码的行为检查。

| 提交 | 原记录指标 | 正文送入模型 | 最终 observation mask=0 | 指标含义 |
| --- | --- | --- | --- | --- |
| Kimi-K3.1-Pro | 0.324667 | 是 | 否 | 混合奖励；格式正确的错答案也得 0.2 |
| GPT-6.1-Sol | 0.321867 | 是 | 是 | 混合奖励；错答案可得 0.1、0.2、0.3 |
| Qwen3.8-Max | 0.329333 | 是 | 是 | 二值 EM；错答案得 0 |
| Gemini-4-Argon | 0.340667 | 是 | 是 | 二值 EM；错答案得 0 |
| Grok-4.7 | 0.370667 | 是 | 是 | 二值 EM；错答案得 0 |
| GPT-6-Astra | 0.389000 | 是 | 是 | 二值 EM；错答案得 0 |

旧评测同时检查最终指标、受保护文件哈希和启动参数，但没有检查最终 mask
或评分函数的调用参数。两条假通过没有修改受保护的 `qa_em_format.py`，而是
在可修改的 `generate_with_search.reward_func` 中启用了
`structure_format_score` 等参数。原来的 `format_score` 是 scorer 保留但没有
用于计算的参数；把它改成 `structure_format_score` 会实际开启
部分奖励。同一个 wrapper 同时用于训练和验证，因此验证均值也被抬高。

这不能作为提高 EM 阈值的证据。例如，在“正确答案得 1、错误答案均得 0.2”
的假设下，15% 正确率也会产生 `0.15 + 0.85 × 0.2 = 0.32` 的平均奖励。
这只是说明指标污染的数值例子，不是对两条历史运行真实 EM 的反推；没有逐条
预测，无法从一个混合奖励均值恢复准确率。

如果仅把阈值调成 0.325，虽然恰好能分开这六个历史数字，但有效修复 Qwen
只剩 0.004333 的余量，等价于 3000 个验证样本中的 13 个；调到 0.33 则会
直接误杀该有效修复。不同指标之间的事后切分不能替代合理校准。已有同口径的
单缺陷对照最高为 0.2237，当前有效修复最低为 0.329333，仍支持保留 0.28。
这些有限运行也不能证明所有随机种子下两类结果都不重叠。

重跑时继续使用 100 rollouts、step 99 和二值 EM ≥ 0.28，同时要求正文、
最终 mask 和评分语义检查全部通过。原两条提交不经修改就重跑，仍应被功能
验收拒绝；若修改了奖励或 mask 后重新训练，应记录为新提交，不能覆盖旧轨迹。
要进一步校准数值阈值，应在同一纯 EM 口径下比较双缺陷、仅修正文、仅修 mask、
两者全修四种配置并重复种子，而不是用受污染的 0.32 分选阈值。

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
re-grade of pulled logs), the legacy hash/launch checks report `skipped_no_workspace`; the mandatory
behavioral check now marks missing source unavailable and blocks a pass.

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
