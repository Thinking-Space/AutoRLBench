# fp8 KV-cache scales are lost across verl's vLLM sleep, so every rollout is "!!!!" to the cap

Training Qwen/Qwen3-4B with verl GRPO on an integer-answer math pool (Skywork-OR1, POLARIS-53K, DAPO-Math-17k) with an fp8 rollout KV cache on vLLM 0.11: verl's level-2 sleep and weight reload zero the KV-cache scales, so every response is `!` repeated to the 16,384-token cap and the reward and gradient are exactly zero from step 1.

| | |
| --- | --- |
| **Author** | Sikuan Yan |
| **Affiliation** | LMU Munich |
| **Type** | Configuration |
| **Domain** | Math RL |
| **Tags** | `fp8-kv-cache` `rollout-engine-state-loss` `sleep-mode` `degenerate-rollouts` |
| **Framework** | verl v0.8.0 (`7aed6b2`, shipped in the workspace), GRPO, vLLM 0.11.0; inspect_ai 0.3.150 installed by `prepare.sh` |
| **Run time** | ~60 minutes (3 steps) |
| **Resources** | 2 × H100 (80 GB) |

## Difficulty

`run_grpo.sh` turns on `kv_cache_dtype=fp8` for rollout throughput. fp8 KV itself works in vLLM 0.11: sampling 64 code prompts × 2 with plain `vllm.LLM` (no verl) gives 39.1% length-capped responses with fp8 vs 35.9% with auto (mean length 11.6k vs 11.2k tokens; code prompts, but the KV-cache path is the same for any prompt, and the point is only that fp8 KV decodes normally without the sleep cycle). The failure comes from the framework: verl v0.8.0 sets `VLLM_SLEEP_LEVEL = 2` for vLLM ≥ 0.8.5 and calls `engine.sleep(level=2)` after rollouts, which frees the weights pool without a backup; on wake-up it pushes the actor's checkpoint tensors through `load_weights()`. vLLM 0.11 keeps the per-layer fp8 scales `Attention._k_scale` / `_v_scale` as GPU tensors in that pool, not as checkpoint weights, so they are never restored. A probe in the image (fp8 KV + sleep mode → generate → `sleep(level=2)` → `wake_up()` → `load_weights()`) shows the scales go from 1.0 to 0.0 and the generation from `'<think>\nOkay, the user is asking for 12 multiplied by 13. …'` to `'!!!!!!!!…'` with Qwen3-4B; the host-side `_k_scale_float` stays 1.0, so nothing in the logs looks wrong.

The symptoms point elsewhere: 16k-token responses and 100% clipping read like "the model cannot stop" or a temperature problem, and the tempting explanation is that fp8 without calibrated scales loses precision, so switching back to bf16 "fixes" it without saying why. It is related to, but distinct from, `infrastructure/verl_quantized_rollout_restart`: there the lost state is the calibration of quantized **weights**, repacked on refresh (vLLM 0.24, whitespace outputs); here the weights are plain bf16 and the lost state is the per-layer fp8 **KV-cache** attention scales, a single rollout flag triggers it, and the same run is healthy on vLLM 0.17. Opening the rollouts shows there is no reasoning at all: 224 of 256 step-1 responses are only `!`; the other 32 also start with `!` and are 87–99.98% `!`, mixed with random multilingual tokens.

| step | recorded (fp8) `critic/score/mean` | control (auto) | fp8 length / clip | auto length / clip | fp8 / auto entropy | fp8 / auto `grad_norm` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.000 | 0.832 | 16384 / 1.00 | 8481 / 0.09 | 0.424 / 0.332 | 0.0 / 0.109 |
| 2 | 0.000 | 0.789 | 16384 / 1.00 | 8561 / 0.12 | 0.404 / 0.325 | 0.0 / 0.129 |
| 3 | 0.000 | 0.777 | 16384 / 1.00 | 7797 / 0.14 | 0.414 / 0.324 | 0.0 / 0.120 |

Horizon: 3 GRPO steps, no in-run validation (`trainer.test_freq=-1`); the evidence is the per-step training metrics (`workspace/math/grpo.jsonl`, `workspace/outputs/grpo/train.log`), three rollout dumps, and the control arm. Both arms ran once (seed 1) in the shared `vllm0.11.0` image on 2 × H100, start model `Qwen/Qwen3-4B` (revision `1cfa9a7`), 32 prompts × 8 samples, response cap 16384, lr 1e-6; fp8 takes ~1155 s per step, auto ~670 s. Single-step diagnostic runs with the same model: fp8 in the `verl-vllm0.17.0` image (response cap 16384) → reward 0.789, clip 0.11. With the response cap lowered to 8192 so that an always-resident vLLM fits next to the actor on 2 GPUs, fp8 with `free_cache_engine=False` (no sleep; actor offload, rollout memory 0.3) → reward 0.488, clip 0.50 (the 8k cap), and none of the 256 responses consists only of `!`, while the identical diagnostic with the sleep left on → reward 0, clip 1.0, 222 of 256 responses only `!`. These are 1-step diagnostics, not controls. Their per-step metric lines, the control's, and the probe and ablation scripts with their outputs are in `evaluations/issue/reference/`. There is no held-out evaluation: the failure is deterministic from the first rollout, before any update, so the rollout dumps are the independent signal. `workspace/math/grpo.jsonl` is the full metric history (the W&B binary was not shipped). The model revision is pinned only as a comment in `prepare.sh`, because the shared `fetch_hf.py` takes no revision.

Issue-only: several equally valid repairs (auto KV, disabling the sleep, re-initialising the scales after each sync, a newer vLLM) restore a healthy run from step 1, and each arm ran once.

## Reference solution

- **issue**: name the fp8 rollout KV cache together with the mechanism in this stack: verl's level-2 vLLM sleep + weight reload loses vLLM 0.11's `_k_scale` / `_v_scale` (they become 0), so attention over the fp8 cache is degenerate and decoding emits `!` to the cap. Naming fp8 with only a generic precision/calibration explanation is partial. Attributions to the model not stopping, the length cap, sampling settings, reward or data, corrupted weight sync, entropy blow-up, or "training did not improve" do not pass.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. As implemented in the shared `judge.py`, passing requires a verdict score of 1.0 and root-cause and evidence scores above 0.8.
