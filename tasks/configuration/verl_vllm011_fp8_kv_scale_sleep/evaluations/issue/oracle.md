# Canonical issue oracle

Reconciled from the task's frozen evidence, the diff between the recorded
workspace and the control run (`kv_cache_dtype=auto`), and diagnostic runs
described in `reference/`.

Expected verdict: **unhealthy**.

## Root cause

- `run_grpo.sh`: `KV_DTYPE=${KV_DTYPE:-fp8}` → `+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8`, combined with verl's vLLM sleep/wake cycle
  - kind: `hparam` (configuration interacting with framework behaviour)
  - observed: the rollout KV cache is fp8. verl v0.8.0 sets `VLLM_SLEEP_LEVEL = 2`
    for vLLM >= 0.8.5 (`verl/third_party/vllm/__init__.py`), and in hybrid mode
    `vLLMHttpServer._sleep_hybrid()` (`verl/workers/rollout/vllm_rollout/vllm_async_server.py`)
    calls `engine.sleep(level=2)` after rollouts. Level 2 frees the weights memory
    pool without a backup; on `wake_up()` verl refills it by pushing the actor's
    checkpoint tensors through `load_weights()`. vLLM 0.11 keeps each attention
    layer's fp8 KV quantization scales as plain GPU tensors in that pool
    (`Attention._k_scale` / `_v_scale`, initialised to 1.0, not checkpoint
    weights), so the sync never restores them and they are 0.0 after wake-up.
    Quantizing / de-quantizing K and V with scale 0 makes attention degenerate,
    and decoding returns token 0 (`"!"`) until the length cap.
  - expected: `kv_cache_dtype=auto` (bf16 KV, no scales), or fp8 KV without the
    level-2 sleep (`actor_rollout_ref.rollout.free_cache_engine=False`), or a vLLM
    version whose scales survive the cycle (vLLM 0.17 in the shared
    `verl-vllm0.17.0` image is healthy with the same run).

A full-credit diagnosis names fp8 KV cache **and** the mechanism that breaks it in
this stack: the KV scales are lost or zeroed by verl's sleep(level=2)/wake-up +
weight-reload cycle (wording such as "the fp8 KV scale buffers are not restored
after verl's weight sync / sleep" is equivalent). Naming fp8 KV cache as the cause
with only a generic precision explanation ("fp8 without calibrated scales loses
accuracy") identifies the right knob but the wrong mechanism and must receive a
partial `root_cause_score`.

## Observable impact

- Primary source metric: `critic/score/mean` (training reward; the recorded run
  has no validation).
- From `workspace/outputs/grpo/train.log` and `workspace/math/grpo.jsonl`
  (3 steps): `critic/score/mean` 0.0, `response_length/mean` 16384.0,
  `response_length/clip_ratio` 1.0, `actor/entropy` 0.424 / 0.404 / 0.414,
  `actor/grad_norm` 0.0 at every step; about 1155 s per step.
- `workspace/outputs/grpo/rollouts/1.jsonl`: 224 of 256 responses are only `!`;
  the other 32 also start with `!` and are 87-99.98% `!`, mixed with random
  multilingual tokens. No
  response contains reasoning or an answer.
- The failure is present at step 1, before any parameter update.
- Control (identical except `KV_DTYPE=auto`): reward 0.832 / 0.789 / 0.777,
  mean length 8481 / 8561 / 7797, clip ratio 0.09-0.14, entropy 0.33 / 0.33 /
  0.32, ~670 s per step.

## Trigger conditions

- vLLM rollout with `kv_cache_dtype=fp8` and no checkpoint-provided KV scales.
- verl hybrid engine with `free_cache_engine=True` (default) and vLLM >= 0.8.5,
  i.e. sleep level 2 plus weight reload on every wake-up.
- vLLM 0.11's `Attention._k_scale` / `_v_scale` stored as GPU tensors in the
  sleep-managed weights pool (not reproduced with vLLM 0.17).
- Not the same object as `infrastructure/verl_quantized_rollout_restart`: the
  model weights here are bf16 and unquantized; the lost tensors are the fp8
  KV-cache attention scales.

## Wrong attributions (must not pass)

- "fp8 KV cache is imprecise without calibration" as the full explanation: plain
  vLLM 0.11 with the same default scale 1.0 and fp8 KV generates normally
  (39.1% vs 35.9% length-capped against auto on the same prompts).
- The model cannot stop / reasoning too long / response cap too small: the
  outputs are a single repeated token from the first position.
- Sampling settings (temperature 1.0, top_p 1.0): identical in the healthy control.
- Reward function or data bug: the reward is 0 because there is no answer; the
  control scores 0.78-0.83 with the same reward and data.
- Corrupted or missing model weights after the weight sync: the same sync works
  with auto KV; only the non-checkpoint scale tensors are lost.
- "Quantized weight scales are lost / weights repacked with wrong scales": the
  weights are unquantized bf16 and load correctly; partial credit at most unless
  the KV-cache scales are named.
- Entropy blow-up / training instability / "training did not improve": the
  failure exists at step 1 before any update, and the gradient is exactly 0.

## Documentation reconciliation

- 100% clip ratio, 16k mean length, the slightly higher entropy (0.40-0.42 vs
  0.32-0.33), zero reward, zero gradient and the ~1.7x step time are all symptoms
  of the degenerate rollouts.
- `reference/kscale_probe.py` (output in `reference/kscale_probe_output.txt`)
  reproduces the mechanism outside verl (with Qwen3-4B): fp8 KV + sleep mode → generate →
  `sleep(level=2)` → `wake_up()` → `load_weights()` → `_k_scale`/`_v_scale` go from
  1.0 to 0.0 and generation becomes `!!!!`.
