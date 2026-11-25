# Canonical issue oracle

Reconciled from the task's frozen evidence and the ladder of recorded arms.

Expected verdict: **unhealthy**.

## Root cause

- `run_grpo.sh` — `actor_rollout_ref.rollout.quantization=fp8` (rollout engine weight
  quantization with a bf16 trainer)
  - kind: `hparam`
  - observed: the vLLM rollout engine runs online fp8-blockwise-quantized weights, and
    every weight sync re-quantizes the trainer's fresh bf16 tensors into the engine
    (`verl/utils/vllm/vllm_fp8_utils.py`, `load_quanted_weights`), so the sampling policy
    `q` keeps tracking a quantized shadow of the trained policy `p`
  - expected: a bf16 rollout (precision matched to the trainer), or any other pairing that
    removes the per-sync re-injection of the quantization mismatch

Mechanism (what a passing diagnosis should articulate): the mismatch between the sampled
policy `q` and the trained policy `p` is small per step but re-injected at every weight
sync, so the drift compounds across steps; the group-centered GRPO advantage has no
corrective term for a bias that lives in the sampler, and the run collapses statistically
— reward slides from ~0.5 to 0.0 over roughly ten steps — with no component failing.

## Observable impact

- Primary source evaluation metric: `critic/score/mean` — healthy through ~step 70
  (0.535 at step 59), 0.27 → 0.18 over steps 70-74, 0.059 at step 75, and 0.0 from step
  78 through step 300 (`outputs/grpo/train.log`).
- The 50-step horizon is a trap: two seeds of this exact config end at 0.584 / 0.514
  (control 0.631 / 0.549) — the collapse is invisible at that budget
  (`reference/ladder_and_control_metrics.txt`).
- Rollouts stay fluent through the collapse (dumped steps 1/59/72/78/84): 0 empty, 0
  `!`-only responses, 27-74% still close an `<answer>` tag, and at step 84 the responses
  still quote the prompt's own numbers ("To create an equation using the numbers 26, 23,
  39, and 20 that equals 97, …"); mean length 1400-1600 chars; entropy ~0.6; no NaN. Much
  later the starved policy degrades into tag-repetition babble (steps 150/300: 509/512
  and 506/512 close an `<answer>` tag around mixed-script fragments; still 0 empty, 0
  `!`-only) — drift, not corruption.
- The bf16 control with the identical seed, data, and 300-step budget stays healthy at
  every step (final 0.678, never below 0.44 after warmup;
  `reference/ladder_and_control_metrics.txt`).
- The fp8-KV arms of the same ladder are a different family: reward 0 from step 1 with
  prompt-blind answers (2/2/0 tagged answers use the prompt's numbers at steps 1/25/50),
  i.e. attention loses the prompt through the uncalibrated KV cache. This run
  (`kv_cache_dtype=auto`, `free_cache_engine=False`) is not that: its rollouts quote the
  prompt's numbers after the collapse.

## Trigger conditions

- A quantized rollout engine paired with a higher-precision trainer, where the weight sync
  path re-quantizes at every step (verl's online fp8 rollout quantization on vLLM), run
  long enough for the drift to compound past the correction capacity of the
  group-centered update (~70 steps at this model/task/lr; shorter budgets look healthy).

## Documentation reconciliation

- The delayed reward collapse is the symptom; the root cause is the quantized-rollout
  configuration, not the training algorithm, data, or reward function (all exonerated by
  the control arms in `reference/ladder_and_control_metrics.txt`).
- The fluent rollouts exonerate generation quality and corruption: this is neither the
  `fp8_kv_scale_sleep` `!`-spam family (no sleep/wake, no KV quantization) nor the
  `quantized_rollout_restart` repack family (weights re-enter the engine from the
  trainer's exact bf16 tensors every sync).
