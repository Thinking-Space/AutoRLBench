# Numerical policy divergence in a completed math GRPO run

DeepSeek-R1-Distill-Qwen-1.5B trained with GRPO + sequence-MIS on the MATH 1460-question subset completes 440 steps in both BF16 and FP16 (one run each), but BF16 reaches a maximum sampled-token probability difference of 0.999926; FP16 reduces the token-weighted mean absolute logprob difference by 8.27× and finishes 2.81 / 2.71 percentage points higher on AIME 2024 / 2025 avg@32.

| | |
| --- | --- |
| **Author** | Yaohui Li |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `numerical-consistency` `hybrid-backend` |
| **Framework** | verl; GRPO + sequence-MIS; torch 2.6.0, vLLM 0.8.5.post1 |
| **Run time** | ~14.3 hours for the BF16 run on 8 H800 GPUs |
| **Resources** | 0 GPU for issue diagnosis; 8 H800 GPUs for the solution scoring replay or manual training reproduction |

## Difficulty

Both runs learn and finish with finite optimizer records. The diagnosis must identify different token probabilities from the synchronized BF16 rollout and learner forward paths, using the pre-update comparison and concrete log evidence. Both use TP=1 / SP=1: a TP=8 explanation does not fit this snapshot. The workspace includes only the BF16 arm; the FP16 comparison is reviewer evidence.

| Recorded measure | BF16 | FP16 |
| --- | ---: | ---: |
| Token-weighted mean absolute logprob difference (nat) | 0.0112137 | 0.00135637 |
| Maximum absolute sampled-token probability difference | 0.9999256 | 0.1971846 |
| Mean batch sequence ESS fraction | 0.0989458 | 0.9506284 |
| Mean sequence-MIS rejected fraction | 0.0564941 | 0.0000266335 |
| Final AIME 2024 avg@32 | 28.6458% | 31.4583% |
| Final AIME 2025 avg@32 | 24.5833% | 27.2917% |

## Reference solution

- **issue**: identify significantly different token probabilities between the BF16 FSDP2 learner forward and vLLM generation forward despite synchronized nominal parameters. Connect the configured forward paths and pre-update comparison to a concrete logprob or probability discrepancy in the frozen logs. Sequence-MIS, ESS and rejection-rate analysis are optional background; a diagnosis need not prescribe FP16, identify an unmeasured CUDA kernel, or know hidden comparator scores.
- **solution**: flip both dtype keys (`actor_rollout_ref.actor.dtype` and `actor_rollout_ref.rollout.dtype`) from `bfloat16` to `float16` in `workspace/verl/verl/trainer/config/math_grpo.yaml`; `float32` for both keys is an equally valid repair under the same reasoning. The minimal patch is `evaluations/solution/oracle/fix.patch`, with the reasoning in `oracle/repair.md`. The scoring replay measures the fix directly: 30 bounded training steps (no validation, no checkpoints), with the token-weighted mean of `parity/logprob_abs_mean_nat` at or below 0.005. The recorded BF16 arm measures 0.0125929 and the FP16 control 0.0015141 over the same 30-step replay, an 8.3x separation on the graded quantity.

## Verification

- **issue**: the canonical LLM judge compares `agent_answer.md` with the hidden `oracle.md`. Scores use verdict / root cause / evidence weights 0.2 / 0.6 / 0.2; passing requires verdict = 1 and both root cause and evidence > 0.8. The instruction and judge Python file remain byte-identical to the family reference. Two deliberate changes are task-local: the verifier entrypoint rejects symbolic-link answers before calling the judge, and the judge prompt scores exactly two evidence requirements (the concrete comparison path and any one correctly interpreted numerical discrepancy). Use `verifier/test.sh` as the grading entrypoint. Real judge API calls were omitted at the user's request; offline checks do not establish LLM grading agreement.
- **solution**: deterministic, no judge. The replay (`verifier/resources/replay.sh`) restores the five pristine pipeline files (the launch scripts and the instrumented trainer/tracking code), wipes `runs/run_001`, and re-runs the submitted workspace's configured training for exactly 30 steps as a foreground process; the agent is told not to launch training itself. The grader (`verifier/tests/grade_submission.py`) then checks the fresh run's structure (30 recorded steps, effective configuration present), the config contract (for example `use_remove_padding` must stay enabled), the pinned assets and files (model, datasets, and the five byte-compared pipeline files), and the graded quantity: the token-weighted mean of `parity/logprob_abs_mean_nat` over the scoring run's training steps must be at or below 0.005. Use `verifier/test.sh` as the grading entrypoint; it needs `AUTORLBENCH_VERIFIER` pointing at the staged verifier tree.

The environment directory matches `qwen3vl_geo3k`: `prepare.sh`, the unchanged shared `fetch_hf.py`, and `asset_pins.json`. Asset pins contain only neutral model/data coordinates and file hashes. Source provenance, label-replacement history and packaging receipts are in hidden `evaluations/issue/resources/source_manifest.json`; the FP16 comparator remains hidden as well. Schema validation, historical scalar preservation and asset-integrity checks pass. Model, data, configuration and reward CPU checks pass in the recorded experiment image.

The first GPU launch check reached model initialization but timed out after multiprocessing reported `AF_UNIX path too long`; the launcher now honors an explicit `TMPDIR`. A CPU tensor-IPC regression reproduces the long-path failure and passes with a short task-owned directory. Job 1492643 passed an 8-H800, two-step training plus one-step resume check: all eight rank restores and 32 checkpoint files were verified, all 96 optimizer audit records were finite, and the observed maximum sampled-token probability difference was 0.380798. The check preserves the formal run's batch, precision and length settings, but skips evaluation and uses save frequency 1. It establishes launch and resume, not a new 440-step result or replay of the step-252 peak. Its receipt is `evaluations/issue/resources/startup_verification.json`.

The pinned single-`FROM` Dockerfile passed a Kaniko metadata-only build with digest-verified registry metadata. A fresh native build in CPU job 1493156 starts its private Docker 28.5.1 daemon but fails before base-image extraction with `remount /, flags: 0x84000: permission denied`; the daemon is then shut down. Full base-image extraction and a public-image container run remain unverified, and the platform restricts external runtime image references. The GPU launch check used the recorded cluster runtime image. Exact limitations and receipts are recorded in hidden `source_manifest.json`, `package_checks.json`, and `native_image_check.json`. The historical runs each completed 440 iterations (1760 PPO minibatch optimizer updates per rank), with a verified eight-rank step-1 restart. No new 440-step training run was launched during packaging.

For a long remote checkout, set `TMPDIR` and `RAY_TMPDIR` to short dedicated paths on the same user storage before running `bash scripts/train.sh`. The benchmark itself stages the workspace at `/workspace`, where the default runtime paths are short.
