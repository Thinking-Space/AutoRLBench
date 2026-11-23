# Collective plan cache omits communicator group and deadlocks CPxDP PPO

Training `openbmb/JustRL-II-base-model` with synchronous PPO on Countdown (`Jiayi-Pan/Countdown-Tasks-3to4`, 8,192 train / 1,024 disjoint test). A cached collective helper omits communicator group identity, deadlocking `CP=2 × DP=2` ranks in a 4-rank NCCL cycle on the first actor update after critic warmup succeeds.

| | |
| --- | --- |
| **Author** | Pan Haoxuan |
| **Affiliation** | ModelBest |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `context-parallel` `data-parallel` `nccl-deadlock` `process-group` `collective-cache` `ppo` |
| **Framework** | `miles` (PPO), `JustRL-II @ 75814318`, PyTorch `2.13.0+cu129`, SGLang `0.5.20.dev59`, Megatron-Core `0.19.0` |
| **Run time** | ~9 minutes (`511 s` wall clock to step-3 NCCL heartbeat termination; healthy 30-step control completes in `629–634 s`) |
| **Resources** | 8 × H100 (80 GB) |

## Difficulty

- **Staged defect vs. upstream**: `miles/utils/collective_plan.py` and its call sites in `ppo_utils.py`, `distributed_utils.py`, and `loss.py` are staged for this task against `JustRL-II @ 75814318` (disclosed in this README and the oracle manifests; omitted from agent-visible `environment/source_manifest.json`), not an unpatched upstream bug.
- **Misleading early health and identical rank control flow**: step-0 evaluation completes (`0.1035` / `0.1045` on 1,024 rows) and critic-only warmup (`NUM_CRITIC_ONLY_STEPS=3`) runs cleanly through steps `0..2` (`train/critic-value_loss`: `0.1221 -> 0.1483 -> 0.0917` on seed 42; `0.1073 -> 0.0933 -> 0.0968` on seed 43). Every collective call site (`get_advantages_and_returns_batch`, `distributed_masked_whiten`, and `compute_ess_ratio_contribution`) runs unconditionally across all ranks with identical tensor shapes, making standard control-flow audits find no skipped collective.
- **Diagonal-rank cache aliasing under `CP=2 × DP=2`**: `miles/utils/collective_plan.py::_signature` keys cached `partial(dist.all_reduce, group=...)` plans by `(backend, world_size, rank_in_group, device.type, device.index, dtype, size_class)` without `group`. Both `distributed_masked_whiten` (`stats_tensor`, 3 elements, `dp_group`) and `compute_ess_ratio_contribution` (`partial_sums`, `[4, 2]` = 8 elements, `cp_group`) map to `"small"`. Once step-3 actor training enters `compute_ess_ratio_contribution`, diagonal ranks `0` `(dp=0, cp=0)` and `3` `(dp=1, cp=1)` hit the cached `dp_group` entry (`DATA_PARALLEL_GROUP`, `SeqNum=2`) while off-diagonal ranks `1` and `2` miss on `rank_in_group` and enter `cp_group` (`CONTEXT_PARALLEL_GROUP`, `SeqNum=3`), hanging on `WorkNCCL(OpType=ALLREDUCE, NumelIn=8)` until the `180000 ms` watchdog aborts.

| arm | step 0 | step 9 | step 19 | step 29 | outcome (`STEPS=30`) |
| --- | ---: | ---: | ---: | ---: | --- |
| defective, seed 42 (shipped) | 0.1035 | — | — | — | step-3 deadlock (`exit=1`, `0` actor steps) |
| defective, seed 43 | 0.1045 | — | — | — | step-3 deadlock (`exit=1`, `0` actor steps, `511 s`) |
| control (`fix.diff`), seed 42 | 0.1016 | 0.2480 | 0.4131 | **0.4453** (`456/1024`) | `exit=0`, `629 s`, `pg_loss=0.0000`, `grad_norm=1.45–2.38` |
| control (`fix.diff`), seed 43 | 0.1006 | 0.2461 | 0.3242 | **0.4150** (`425/1024`) | `exit=0`, `634 s`, `pg_loss=0.0000`, `grad_norm=1.51–2.57` |

## Reference solution

- **issue**: verdict `unhealthy`; identify `miles/utils/collective_plan.py::_signature` omitting communicator `group` from the `_plans` cache key so diagonal `CP=2 × DP=2` ranks alias `distributed_masked_whiten` (`dp_group`, `numel=3`) with `compute_ess_ratio_contribution` (`cp_group`, `numel=8`) in `"small"` at actor step 3; cite step-0/critic warmup success followed by `WorkNCCL(NumelIn=8, Timeout(ms)=180000)` on `DATA_PARALLEL_GROUP` vs `CONTEXT_PARALLEL_GROUP`. Reject attributions blaming conditional empty-shard skips, slow networking or `--distributed-timeout-minutes`, SGLang weight sync, or Countdown reward parsing.
- **solution**: add `group` into the tuple returned by `_signature(tensor, group)` in `miles/utils/collective_plan.py` (`evaluations/solution/oracle/fix.diff`), keeping `dp_group` and `cp_group` plans distinct.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted `0.2 / 0.6 / 0.2`. Passing requires `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8` (a weighted score `>= 0.8` alone does not pass if either `root_cause_score <= 0.8` or `evidence_score <= 0.8`).
- **solution**: the agent submits `report.md` and does not train; the harness reruns `run_justrl2_ppo.sh` (`STEPS=30`, `SEED=42`) per `run_spec.json` and grades only the fresh run via `evaluations/solution/verifier/tests/grade_submission.py`. Pass requires `heldout/countdown_exact_acc@29 >= 0.32` (re-scored from `experiment_logs/val_generations/29.jsonl`, `step == 29`, `dataset == "countdown"`, zero malformed rows, exact `1,024` unique held-out test instances matched against `data/countdown/test.jsonl`), continuous rollout steps `0..29` with exact actor updates `3..29` (`train/pg_loss`), critic updates `0..29` (`train/critic-value_loss`), and step-29 `eval/countdown` in `metrics.jsonl` matching the re-scored dump accuracy (`<= 1e-6`), plus pinned `run_justrl2_ppo.sh`, dataset/script SHA-256, and JSONL split integrity. Calibrated across `4` single-node `8 × H100 (80 GB)` runs (`2` defective, `2` control) and replayed by hardened `grade_submission.py` inside `autorlbench-justrl2@sha256:8d2eb1fe...` (`stage_integrity=true` with `pyarrow`): `0.32` sits `+0.3200` above the defective arms (no step-29 evaluation reached), `+0.2184` / `+0.2194` above control step 0 (`0.1016` / `0.1006`), and `-0.0950` below the lower control (`0.4150`). In addition, a fresh cold end-to-end `8 × H100` run from empty `models/` and `data/` (`CUDA_VISIBLE_DEVICES="" environment/prepare.sh` -> `fix.diff` -> first-time GPU `convert_hf_to_torch_dist.py --attention-backend fused --no-gradient-accumulation-fusion` -> 30-step PPO -> real `verifier/test.sh`) scored `0.537109375` (`550/1024`, `passed=true`, `score=1.0`, `stage_integrity=true`). *Verification scope*: executed on `wuxi_train` (`speech-wx`) `8 × H100 (80 GB)` inside `autorlbench-justrl2@sha256:8d2eb1fe...` with SHA-256-verified local HF caches (batch pods on this cluster cannot reach `huggingface.co` over outbound internet); `HarborTaskMaterializer` passed locally, while full local Docker Harbor runtime and live LLM judge API calls in `evaluations/issue/verifier/judge.py` (tested only with mocked boundary scores) were not executed.
