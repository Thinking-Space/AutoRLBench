# Stale whitening moment cache corrupts PPO advantages on empty context-parallel shards

Training `openbmb/JustRL-II-base-model` with synchronous PPO on Countdown (`Jiayi-Pan/Countdown-Tasks-3to4`, 8,192 train / 1,024 disjoint test). A mutable zero-origin tensor cached per device aliases the tensor returned for empty local ZigZag CP=2 shards and is mutated in place by `dist.all_reduce`, silently compounding stale moments across steps without crashing or raising NaNs and degrading step-29 accuracy from `0.4170` / `0.3447` down to `0.2129` / `0.0342`.

| | |
| --- | --- |
| **Author** | Pan Haoxuan |
| **Affiliation** | ModelBest |
| **Type** | Algorithm |
| **Domain** | Math RL |
| **Tags** | `context-parallel` `advantage-whitening` `stale-state-cache` `silent-divergence` `ppo` |
| **Framework** | `miles` (PPO), `JustRL-II @ 75814318`, PyTorch `2.13.0+cu129`, SGLang `0.5.20.dev59`, Megatron-Core `0.19.0` |
| **Run time** | ~11 minutes (`639–642 s` for 30 steps; healthy control completes in `628–636 s`) |
| **Resources** | 8 × H100 (80 GB) |

## Difficulty

- **Staged defect vs. upstream**: `miles/utils/moments.py` and its per-group DP whitening wiring in `distributed_utils.py` and `loss.py` are staged for this task against `JustRL-II @ 75814318` (disclosed in `environment/source_manifest.json`), not an unpatched upstream bug.
- **No crash, no NaNs, and a misleading early lead**: all 30 rollout, actor, critic, and evaluation steps finish cleanly (`exit_code = 0`). At `eval@9`, the defective run actually leads the control on **both** seeds (`0.2910` vs `0.0566` on seed 42; `0.2324` vs `0.2188` on seed 43), and on-policy health metrics stay pristine (`train/ess_ratio = 1.0`, `train/ppo_kl = 0.0`, `train/pg_clipfrac = 0.0`, `train/train_rollout_kl ≈ 0.0011`, `train/critic-value_loss = 0.0187–0.1417`).
- **Cached buffer mutation on empty CP shards**: `miles/utils/moments.py::MomentReducer` caches `self._origins[(values.device, torch.float32)]` (a 2-element zero buffer) and passes it as the initializer to `functools.reduce(add, blocks, origin)`. Inside `distributed_masked_whiten` (invoked per 4-sample prompt group in `compute_advantages_and_returns`), short Countdown responses (`response_len <= prompt_len / 3`) leave CP rank 1's ZigZag local shard empty (`values.numel() == 0`), so `reduce` returns the cached `origin` tensor object itself. In-place `dist.all_reduce(moment_tensor, group=process_group)` permanently overwrites `self._origins` with that group's reduced `[sum, sum_sq]`, which compounds on every subsequent empty-shard reduction while contributing `0` to `count_tensor`, driving `train/pg_loss` to `-3070.99` (`train/grad_norm` to `9669.23`) and truncation ratio up to `0.7275` / `0.7656`.

| arm | step 0 | step 9 | step 19 | step 29 | `trunc@29` (`len@29`) | `train/pg_loss` | `train/grad_norm` |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| defective, seed 42 (shipped) | 0.1006 | 0.2910 | 0.2666 | **0.2129** (`218/1024`) | 0.7275 (`416.9`) | `0.0 -> -3070.99 -> -345.71` | `2.82 -> 9669.23 -> 2525.50` |
| defective, seed 43 | 0.0977 | 0.2324 | 0.1309 | **0.0342** (`35/1024`) | 0.7656 (`426.1`) | `-101.49 -> -679.17 -> -599.91` | `793.76 -> 3440.76 -> 1490.59` |
| control (`fix.diff`), seed 42 | 0.0996 | 0.0566 | 0.3564 | **0.4170** (`427/1024`) | 0.5205 (`318.1`) | `0.0000` | `1.44–4.42` |
| control (`fix.diff`), seed 43 | 0.0977 | 0.2188 | 0.2822 | **0.3447** (`353/1024`) | 0.6152 (`402.4`) | `0.0000` | `1.18–2.98` |

## Reference solution

- **issue**: verdict `unhealthy`; identify `miles/utils/moments.py::MomentReducer` caching a mutable zero origin tensor per `(device, torch.float32)` and returning it from `functools.reduce` on empty local ZigZag CP shards, where in-place `dist.all_reduce` in `distributed_masked_whiten` mutates and compounds stale moment statistics without raising NaNs or crashing; cite step-29 collapse (`0.2129` / `0.0342` vs `0.4170` / `0.3447`), `train/pg_loss` / `train/grad_norm` explosion (`-3070.99` / `9669.23`), and unchanged `ess_ratio = 1.0`, `ppo_kl = 0.0`, `critic-value_loss`. Reject attributions blaming collective desync, critic regression, hyperparameters (`lr=2e-6`, `gae_lambda_k=0.5`), SGLang weight sync, or Countdown reward parsing.
- **solution**: remove `self._origins` caching and allocate a fresh `origin = torch.zeros(2, device=values.device, dtype=torch.float32)` per `MomentReducer.__call__` invocation (`evaluations/solution/oracle/fix.diff`).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted `0.2 / 0.6 / 0.2`. Passing requires `>= 0.8`.
- **solution**: the agent submits `report.md` and does not train; the harness reruns `run_justrl2_ppo.sh` (`STEPS=30`, `SEED=42`) per `run_spec.json` and grades only the fresh run via `evaluations/solution/verifier/tests/grade_submission.py`. Pass requires `heldout/countdown_exact_acc@29 >= 0.28` (re-scored from `experiment_logs/val_generations/29.jsonl`, `>= 1,000` rows), continuous rollout steps `0..29` with `>= 27` actor steps and step-29 `eval/countdown` in `metrics.jsonl`, plus pinned `run_justrl2_ppo.sh`, dataset/script SHA-256, and JSONL split integrity. Calibrated across `4` single-node `8 × H100 (80 GB)` runs (`2` defective, `2` control) and replayed by `grade_submission.py` inside `autorlbench-justrl2@sha256:8d2eb1fe...`: `0.28` sits `+0.0671` above the higher defective run (`0.2129`, and `+0.2458` above seed 43 `0.0342`), `+0.1804` / `+0.1823` above control step 0 (`0.0996` / `0.0977`), and `-0.0647` below the lower control (`0.3447`). *Verification scope*: executed on `wuxi_train` (`speech-wx`) `8 × H100 (80 GB)` inside the exact digest-pinned container image using staged Countdown data/weights and `0`-GPU container replay of `grade_submission.py`; local Harbor/Docker end-to-end `prepare.sh` execution was not run in this environment.
