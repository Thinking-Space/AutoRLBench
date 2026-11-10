# Native wrong expected outputs in the training labels silence correct solutions

Training Qwen3-4B-Instruct-2507 with GRPO on TACO at matched `EASY`/`MEDIUM`
difficulty, but ~40% of the training rows are problems whose expected outputs
are natively wrong in the dataset: the problems' own bundled solutions agree
with each other on an output and still fail their own tests. Under the binary
all-tests-pass reward, a rollout that solves such a problem correctly is still
scored 0, so GRPO teaches the model to discard correct algorithms for those
problems. Training reward is pinned below the clean baseline scaled by the
clean fraction, the externally-correct-but-scored-0 rate persists instead of
vanishing, and held-out accuracy regresses instead of improving.

Constructed from the label-noise brief
`raw_tasks/task_briefs/label_noise_1_taco_wrong_expected_output.md`
(family: 标签原生有误 — the grading reference is wrong in the source dataset
itself, a false-negative channel; nothing is flipped by hand).

| | |
| --- | --- |
| **Author** | Yinghao Chen |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | Data |
| **Domain** | Code RL |
| **Tags** | `label-noise` `wrong-expected-output` `false-negative` |
| **Framework** | verl (GRPO) + vLLM 0.17 / torch 2.10, Qwen3-4B-Instruct-2507 on TACO |
| **Run time** | ~10.4 hours (49 steps recorded on a contended host; the clean control at the shipped 40 steps took ~2.1 h on dedicated GPUs) |
| **Resources** | 4 × H800 (80 GB) |

Dataset: TACO train, ~40% unverified dirty rows (wrong-expected-output family) +
~60% verified-clean rows, `EASY`/`MEDIUM` matched; execution-verified
`EASY`/`MEDIUM` held-out from TACO's test split (184 rows). Both arms recorded
2026-11-10 (defective through step 49 of 50, control at the shipped 40 steps);
numbers in `evaluations/solution/oracle/control_reference/historical_metrics.json`.

## Status: rebuilt around Qwen3-4B-Instruct-2507 (2026-11-09), both arms recorded

The task was first calibrated on Qwen2.5-1.5B-Instruct, then rebuilt on a
stronger model: with the 1.5B, both arms improved within noise on the held-out
split (3.3% -> 3.8%), leaving the capability-degradation channel invisible;
the user rejected that calibration. Model selection probe on the frozen
184-row held-out split (greedy, then sampled at the training validation's
temperature 0.7 / top_p 1.0 / n=4, taco.py binary scoring):

| Candidate | greedy | sampled (t0.7, n=4) | note |
| --- | --- | --- | --- |
| Qwen3-1.7B (thinking off) | 3.8% | — | far below bar |
| Qwen3-4B (thinking off) | 12.0% | — | below bar |
| **Qwen3-4B-Instruct-2507** | 31.5% | **30.3%** (4096 max tokens) | chosen |

Response-length sweep for the chosen model (sampled, same settings):
4096 -> 30.3% acc / 20.1% truncated; 8192 -> 31.8% / 12.6%;
**16384 -> 33.8% / 2.3%**. Truncated rollouts NEVER pass their tests
(P(correct | truncated) = 0.000 at every budget), so the launch script's
`MAX_RESPONSE_LENGTH` is set to 16384 (measured 2026-11-09; the 1.5B-era
4096 wasted a fifth of every rollout batch). The recorded-run baseline for
the chosen model is 33.8% sampled at 16384.

Remaining before this task can move to `tasks/`:

1. ~~**Run the purification** and commit the selection inputs~~ **DONE
   2026-11-06**: `selection/dirty_manifest.jsonl` (552 problems),
   `selection/test_manifest.jsonl` (292 held-out problems, EASY+MEDIUM) and
   the grader's `corrected_outputs.json` (552 entries) are committed; the
   audit receipts live outside the repo (see the data_prep README).
2. **Record the buggy run on the new model**: DONE 2026-11-10 through step 49
   of 50 (a PREEMPTABLE-pool reaper killed the pod during step-50/final
   validation; log tail shows a healthy 48/50 at kill time, so this is a
   platform kill, not a training error, and the 49 recorded steps + all
   validations 0-45 are complete). Result: held-out accuracy flat at the
   step-0 level over validations 0-45 (0.353 -> 0.348, mean 0.343, no trend)
   — no learning on the held-out split, consistent with the defect.
   Training reward mean 0.370 (range 0.09-0.65). Correct-but-zero rate over
   dumps 36-40 (the 40-step grader window): per-step 0.367 / 0.195 / 0.281 /
   0.320 / 0.141, metric (max) = 0.3672 — the defect channel is loud and
   does not fall with training.
3. **Record the control arm on the new model**: DONE 2026-11-10 at the
   reduced budget of **40 steps** (the task's run_spec/instruction/grader
   all say 40), on a NORMAL-priority devspace: same launch config, train =
   1,380 verified-clean rows (EASY 1,100 / MEDIUM 280, same seed 20261106,
   linux sha256 `887ef2f2…` verified), test.parquet byte-identical. Result:
   grader metric 0.0000 over dumps 36-40 (structural 0, as designed),
   training reward mean 0.602 (last five 0.627), held-out accuracy 0.321 ->
   0.359 — the control arm learns where the defective arm stagnates.
4. **Calibrate** the grader spec from the two runs: DONE —
   `historical_final_baseline` = 0.3671875 (defective arm's dumps 36-40
   max), `required_final_score_max` = 0.02 confirmed (defective 0.3672 vs
   control 0.0000); `evaluations/solution/oracle/control_reference/`
   refreshed with `historical_metrics.json` (+ `group_advantage.json` for
   the dirty-group advantage split).
5. Re-read the packaging checklist in `skills/package-task/SKILL.md`.

## The defect

`workspace/taco_verl/train.parquet` is assembled by `environment/prepare.sh`
from two public sources:

* **clean rows**: problems in `likaixin/TACO-verified` (every kept problem
  has a bundled solution that passes its own tests within 10 s);
* **dirty rows**: train problems NOT in that set (the ~12,545-problem dirty
  pool, 49.31% of TACO train), further purified to the wrong-expected-output
  family: >= 2 textually distinct bundled solutions run the problem to
  completion, fail the SAME test, with IDENTICAL **non-empty** actual outputs
  -- and no bundled solution passes under a loose 30 s budget. The non-empty
  requirement is load-bearing: when every solution produces an empty output
  the "agreement" establishes nothing about the expected output being wrong
  (that pattern is the unparseable-test-input family -- GfG-style
  `N = 8\narr[] = {...}` test inputs no stdin solution can read -- and an
  empty-vs-empty match would vacuously confirm any print-nothing response
  downstream). The purification
  receipts (per problem: failing test index, the dataset's expected output,
  the agreeing solutions' output) are `environment/selection/dirty_manifest.jsonl`.

Mix (frozen 2026-11-06, seed 20261106): 40% dirty / 60% clean at matched
`EASY`/`MEDIUM` difficulty (the brief's 30-50% band, aligned with
AceReason-Nemotron §4.3.6's injection ratio), **1,380 train rows (552 dirty
= 40.0%, EASY 1,100 / MEDIUM 280; max prompt 1,884 tokens; every manifest
problem passes the 2,048-token filter, so no dirty row is lost to it)**. The
dirty manifest holds 552 confirmed problems (EASY 440 / MEDIUM 112) purified
from the complete 3,489-problem EASY/MEDIUM stdin candidate pool (verdicts:
552 native-wrong-expected-output under the strict non-empty criterion,
1,371 unverified -- predominantly the unparseable-test-input family,
382 passes_loose, 202 disagree, 13 timeout/compile/error); all 552 are
sampled in. The held-out
`test.parquet` (184 rows, `EASY`/`MEDIUM`, stdin) is drawn from TACO's own test
split (problem_id `test{N}`, as in the sibling task's parquets; the split's
EASY band alone is too small for 184 rows) and restricted to
problems whose own bundled solutions pass their tests under the loose budget
(`selection/test_manifest.jsonl` -- the verified subset covers train rows
only, so held-out cleanliness is established by this task's own execution
receipts).

Everything else is the healthy baseline, identical to the sibling task
`tasks/data/verl_taco_difficulty_mismatch` except the model, the response
budget and the launch-script name: same
`verl/utils/reward_score/taco.py` (sandbox execution, 1 only when every
test passes), same hyperparameters. The reward is not modified -- the defect is
what it is fed.

Why this difficulty band and this model: TACO `EASY`/`MEDIUM` with a model
that solves a meaningful fraction of problems is what activates the
false-negative channel: a problem the model never solves produces no
correct-but-punished rollout, and the defect degenerates into "data too
hard" (the sibling task's failure). Qwen3-4B-Instruct-2507 solves 33.8% of
the held-out split sampled at the training budget -- 2.75x the 1.5B's ~12%
clean-`EASY` step-0 rate -- and is a pure non-thinking instruct model, so
the reward's first-fenced-block extraction never sees thinking-mode
scratchpad code. (Hybrid Qwen3 with thinking disabled solves only 12.0%
-- the template's empty think-block is not why; the 2507 instruct
post-training is what carries the capability.)

## Difficulty

Nothing fails loudly: the run completes every step with valid shapes, the
training reward is nonzero and noisy (0.09-0.65 per step, mean 0.370), and
held-out evaluation keeps working — a noisy-but-alive run, not a crash. The
reward's noisiness even mimics healthy exploration. What is wrong requires
comparing scorers, not watching curves: on a dirty problem the run's reward is
0 while executing the very same response produces exactly the output the
problem's own agreeing bundled solutions produce. That is the single
strongest piece of evidence, readable from the bundled dumps alone.

The mechanism, from the recorded runs (both arms 2026-11-10):

* **Correct-but-scored-0 rate** (external judge over the run's dumps
  36-40, the grading window): 47/25/36/41/18 of 128 rows per dump on the
  defective arm (grader headline 0.3672) versus 0/128 on every dump of the
  clean control (0.0000). The two scorers separate on the dirty subset,
  and only on it — the defect's fingerprint.
* **Training reward pinned at the clean-share baseline**: defective mean
  0.370 versus the control's 0.602 x 0.6 clean share = 0.361 — the dirty
  40% contributes nothing learnable, so the mean cannot track the clean
  problems' own learnability.
* **Advantage structure on dirty groups** (dumps 36-40): 27 of 46 dirty
  GRPO groups contained an externally-correct-but-scored-0 rollout (167
  rows); 164 of the 167 received zero advantage (the whole group scored 0)
  and 3 received negative advantage. The correct-algorithm learning signal
  is removed on exactly those problems.
* **Held-out accuracy stagnates instead of improving** (184 rows, binomial
  SE ~ 0.035; one run per side):

| val step | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| defective `acc` | 0.353 | 0.332 | 0.293 | 0.359 | 0.332 | 0.364 | 0.353 | 0.359 | 0.337 | 0.348 |
| control `acc` | 0.321 | 0.321 | 0.380 | 0.332 | 0.364 | 0.337 | 0.337 | 0.353 | 0.359 | — |

## Recorded run

* Log: `workspace/logs/qwen3_4b_instruct_2507_taco_fsdp_20261110_012754.log`
* 4 GPUs, 49/50 steps in ~10.4 h on a contended host (the PREEMPTABLE
  devspace was reaped during step-50/final validation; the log tail shows a
  healthy 48/50 at kill time — a platform kill, not a training error, and
  the 40-step grading window is fully covered with margin).
* Config: `train_batch_size=16`, `rollout.n=8`, `lr=1e-6`,
  `MAX_RESPONSE_LENGTH=16384`, rollout temperature 1.0, validation
  temperature 0.7, validation every 5 steps.
* `workspace/rollout_data/` holds the 49 per-step training rollout dumps
  (128 rollouts each) and `workspace/validation_data/` the ten validation
  dumps (184 rollouts each) — the primary quantitative evidence for the
  causal chain.

## Buggy run and judging evidence

* The task-root `run_spec.json` rerun is the buggy arm: same launch script,
  40 steps (validation every 5), rollout dumps per step under
  `experiment_logs/rollout_data/` (128 rollouts = 16 prompts x 8).
* Frozen evidence shipped in `workspace/`: all 49 rollout dumps and 10
  validation dumps (0, 5, ..., 45) of the recorded defective run with each
  row's `gts` field dropped, plus the run's training log — sanitized (paths,
  host, IP, username; replacements and per-file sha256 before/after in
  `environment/source_manifest.json`, no metric value touched). The 49-step
  evidence covers the 40-step grading window with margin.
* The solution grader's external judge: for every dumped response with score 0
  on a dirty problem, execute the response's code on the problem's recorded
  failing test and compare with the corrected output. "Correct solution
  scored 0" samples are the metric; max rate over the last five step dumps.
  Anchoring: launch script and `taco.py` byte-compared against pristine
  verifier copies, `test.parquet` sha-pinned, `train.parquet` floored at
  `min_train_rows` and checked disjoint from the test prompts, run-reality =
  the dump sequence 1..40 itself.
* Issue tier: LLM judge against the hidden oracle (verdict / root cause /
  evidence = 0.2 / 0.6 / 0.2, conjunctive pass gate), as in the sibling task.

## Reference solution

Remove the natively-wrong-expected-output rows from `taco_verl/train.parquet`
(or replace them with verified-clean rows at matched `EASY`/`MEDIUM`
difficulty). The repaired `train.parquet` must keep the taco_verl schema,
prompt template, and `data_source` unchanged, keep at least
`min_train_rows` rows, and stay disjoint from `test.parquet` prompts;
`test.parquet`, the launch script, and `verl/utils/reward_score/taco.py`
must remain byte-identical. Identifying the dirty rows offline is part of
the task: a row's own tests can be executed against candidate solutions
(the workspace model can generate candidates, and the worked examples in
each problem statement cross-check its test cases).

The recorded control arm (same launch config, same seed, 1,380
verified-clean rows, linux sha256 `887ef2f2…`) is the calibration anchor:
correct-but-zero rate 0.0000, training reward mean 0.602, held-out
accuracy 0.321 -> 0.359.

## Verification

* **issue**: judge.py over `$AUTORLBENCH_RESULTS/agent_answer.md` vs
  `evaluations/issue/oracle.md`. No GPU needed. The oracle's numbers are
  filled from the recorded defective run.
* **solution**: `grade_submission.py` re-scores the fresh run's rollout dumps
  (see above). Calibration on the new model (2026-11-10, one run per side,
  margins reported without variance): defective arm 0.3672 FAIL vs control
  0.0000 PASS at `required_final_score_max` = 0.02 (margin 0.347 to the
  upper bound, 0.02 to the lower). Full numbers in
  `evaluations/solution/oracle/control_reference/historical_metrics.json`.

## Packaging notes

**Assets.** All fetched at prepare time from public HF repos, cache-first via
the task-local `fetch_hf.py` (with one deviation: `--allow` restricts
BAAI/TACO to the `ALL/train-*.parquet` shards -- the repo's loading-script
`train/`+`test/` arrow duplicates are ~8 GB this task never reads):

| Asset | Repo | Note |
| --- | --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` | non-thinking instruct; ~7.6 GB |
| Native pool | `BAAI/TACO` @ `d593ed0a2becbbc952230bb89be09189bf1056dc` (ALL/train, 9 shards, ~2.4 GB) | 25,443 rows |
| Held-out split | `BAAI/TACO` @ `d593ed0a2becbbc952230bb89be09189bf1056dc` (ALL/test, 1 shard, ~246 MB) | 1,000 rows; the sibling task's `test{N}` id scheme verified positionally against it |
| Verified subset | `likaixin/TACO-verified` @ `9e68575e01938f016199eda6f07d1a7ce96b35e1` (`taco_verified.json`, 1.9 GB) | 12,898 rows, id <-> row mapping verified by question text |

Both dataset repos are pinned (`--revision` in prepare.sh, see fetch_hf.py's
docstring): TACO's contents drift upstream -- the sibling task's published
test.parquet already disagrees with the current ALL/test split (different
expected outputs, different test counts), so "fetch main" is not reproducible.
The model is pinned for the same reason. The pin plus the manifest's
per-problem question_sha256 cross-check makes the assembly fail loudly rather
than silently mix rows from a drifted revision.

**Assembly reference (frozen, seed 20261106, Qwen3-4B-Instruct-2507
tokenizer).** `assemble_dataset.py` run against the pinned revisions
reproduces the frozen selection exactly: 1,380 train rows (552 dirty,
EASY 1,100 / MEDIUM 280) and 184 test rows. Row membership is unchanged
from the Qwen2.5-1.5B freeze (same problem_ids in the same order; the
2,048-token filter drops nothing on either side: max train 1,884 / max test
1,369 Qwen2.5 tokens); only the `extra_info.prompt_tokens` metadata changed
(the Qwen3 chat template renders ~21 fewer tokens per prompt). The parquet
**byte** hashes are platform-dependent: pyarrow bundles a different snappy
build per platform, so the same logical table serializes to different bytes
on macOS (author machine) and linux x86_64 (the docker image the
prepare/verifier stages run in). Reference values (linux x86_64, canonical):

| | linux x86_64 (runtime, canonical) |
| --- | --- |
| `train.parquet` | `1eaf778569fd1238dd397e61911c5925f105b14e6f5a3e380c2347e270d7724b` (328,576,218 B) |
| `test.parquet` | `b869d3d819b87751920a5af9012af36776db78d85b4e11602f8a228ec00b866a` (55,266,146 B) |
| control `train.parquet` | `887ef2f251dcd2368487c1bac687c4dd7a9f8348ce5d243a28055b07915bd6c1` (97,509,436 B; author-side clean arm, not assembled by prepare.sh) |

The grader spec pins the linux values (the verifier runs in the image).
A rerun on linux that disagrees with the linux hash means the tokenizer
snapshot or a source drifted. The `Qwen/Qwen3-4B-Instruct-2507` tokenizer
used for `prompt_tokens` is pinned with the model revision above.

**Assembly is deterministic** given the same upstream revisions and the
committed `selection/` inputs (seeded sampling, stable shard order), so the
parquets do not need to be shipped. Alternative once calibrated: publish the
assembled parquets as a neutral-named HF archive and collapse prepare.sh to
plain fetches (the sibling task's hardening path).

**Known leak surface.** The agent does not see `environment/` (harness
`sees_environment_dir`), so the dirty manifest, the mix spec and prepare.sh's
logic are agent-invisible. The workspace `train.parquet` itself is the defect
and is agent-visible by design (diagnosis material). No known leak beyond
that; re-check after any harness change.

**Issue-only sibling note.** `tasks/data/verl_taco_difficulty_mismatch` is the
difficulty-mismatch sibling built on the same stack; its answer is listed as
a wrong attribution here, and vice versa.

**Pinned files.** The launch script and `taco.py` pristine copies under
`evaluations/solution/verifier/resources/` are byte-copies of the workspace
files as of this draft; re-sync them if the workspace versions change before
calibration.
