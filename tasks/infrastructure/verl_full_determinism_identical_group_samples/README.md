# Full determinism makes every GRPO group identical

Training Qwen2.5-1.5B-Instruct with GRPO on GSM8K with verl's full-determinism mode turned on. Every prompt's 8 rollout samples come back byte-identical, so every group has zero reward variance and zero advantage, and the policy never changes, while sampling, reward and logging all look normal.

| | |
| --- | --- |
| **Author** | Jianing Yin |
| **Domain** | infrastructure |
| **Tags** | `verl` `grpo` `vllm` `determinism` `sampling-seed` |
| **Model** | Qwen/Qwen2.5-1.5B-Instruct |
| **Dataset** | openai/gsm8k main, 7473 train / 1319 test |
| **Framework** | verl @ `6f2aa24` as released, GRPO, `run_grpo_fd.sh` |
| **Environment** | `verl` · 1 GPU |
| **Evaluations** | issue · solution |
| **Recorded runs** | 2 defective + 2 repaired runs, 30 steps each, 1 GPU (~20 min each), plus 4 three-step restart runs |

## Difficulty

With `actor_rollout_ref.rollout.full_determinism=True`, `vLLMHttpServer.generate` in `verl/workers/rollout/vllm_rollout/vllm_async_server.py` sets `sampling_params.setdefault("seed", self.replica_rank + self.config.seed)`, so every request on a replica gets the same seed. Full determinism also turns on vLLM's batch-invariant mode, which makes the logits of identical prompts identical regardless of batching. The `n` copies of a prompt therefore draw the same random stream from the same distribution and return the same tokens. verl's GRPO computes each group's reward standard deviation as 0, every advantage is 0, and the policy gradient vanishes. The workspace is verl `6f2aa24` without modification; current verl main has the same line, and verl's own determinism tests run with `rollout.n=1`, which cannot show the problem.

Nothing fails. Rollouts are sampled at temperature 1.0 and `actor/entropy` stays between 0.33 and 0.68, so "greedy decoding" is wrong. Each group has its 8 rows under one uid, so this is not the singleton-group defect of `verl_grpo_uid_after_repeat`. Training reward moves from batch to batch (0.06 to 0.27) because the prompts change. What gives it away is that `critic/advantages/max` and `/min`, `actor/pg_loss`, `actor/grad_norm` and `actor/kl_loss` are exactly 0 at every one of the 30 steps, held-out accuracy is the same 0.08719 at all 7 validations, and in `outputs/rollouts/` the 8 responses of every prompt are identical.

| `val-core/openai/gsm8k/acc/mean@1` | step 0 | step 5 | step 15 | step 30 | groups with 8 identical responses |
| --- | ---: | ---: | ---: | ---: | --- |
| defective, data seed 1 / 2 | 0.0872 / 0.0872 | 0.0872 / 0.0872 | 0.0872 / 0.0872 | 0.0872 / 0.0872 | 64 of 64 at every step |
| repaired, data seed 1 / 2 | 0.0872 / 0.0872 | 0.6543 / 0.6657 | 0.7180 / 0.7271 | 0.7384 / 0.7354 | 0 of 64 at every step |

The repaired arm differs from the defective one only in the seed injection (`evaluations/solution/oracle/reference.patch`). A restart check with a probe on `generate()` (`evaluations/issue/reference/evidence.md`) shows that the repaired run keeps the guarantee full determinism is for: run twice, all 1536 sampled requests of 3 steps get the same seeds and the same tokens, and the rollout dumps are identical line for line.

## Reference solution

- **issue**: name the per-request seed injection in `vLLMHttpServer.generate` (`replica_rank + rollout.seed` for every request) as the root cause, explain that with batch-invariant vLLM it makes the `n` samples of a prompt identical, and connect it to zero group variance, zero advantages and an unchanged policy. Attributions to the learning rate, KL loss, reward function or answer format, temperature or greedy decoding, uid grouping, or "full determinism is incompatible with GRPO" do not pass.
- **solution**: derive the sampling seed per sample from something that differs between the `n` copies of a prompt and is stable across restarts. The reference patch uses the sample's row in the batch (`priority`, which `AgentLoopManager` assigns before dispatch) together with a hash of the prompt tokens; the request id cannot be used because it is a fresh uuid4 per call. Turning full determinism off, seeding from a random source, or changing `n` or the replica count avoids the defect instead of fixing it; the rerun pins full determinism, one replica and the base seed, and checks that the run is still reproducible.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle and scores
  verdict, root cause and evidence. As implemented in the shared `judge.py`, a run passes only if
  the verdict score is 1.0 and the root-cause and evidence scores are both strictly greater than 0.8;
  the 0.2 / 0.6 / 0.2 weighted score is reported but does not decide the pass.
- **solution**: the agent submits `report.md` and does not train; the harness reruns the
  experiment per `run_spec.json` (the 30-step run, then the same command for 3 steps under
  `experiment_logs/repeat/`) and grades only those runs. Three checks must all pass:
  `val-core/openai/gsm8k/acc/mean@1` at step 30 ≥ 0.4, with the verifier's own re-grade of the
  step-30 validation dump against a protected answer key agreeing within 0.002; at every training
  step fewer than half of the 64 groups may have 8 identical responses; and the repeated run's
  rollout dumps for steps 1 to 3 must equal the main run's. Margins: 0.313 above the defective arm
  (0.0872 everywhere), 0.335 below the lower repaired step 30 (0.7354), 0.254 below the lowest
  repaired point from step 5 (0.6543). Calibrated on 2 runs per side.
