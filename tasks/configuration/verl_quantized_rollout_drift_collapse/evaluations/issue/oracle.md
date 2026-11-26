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
  (0.535 at step 59; the run's actual peak is 0.635 at step 45), 0.27 → 0.18 over steps 70-74,
  0.059 at step 75, first 0.0 at step 78 (afterwards only isolated 1-3-of-512
  single-sequence successes and a brief partial recovery up to ~0.07 through step 145;
  exactly 0.0 from step 146 through step 300) (`outputs/grpo/train.log`).
- Rollouts stay fluent through the collapse (dumped steps 1/59/72/78/84): 0 empty, 0
  `!`-only responses, 27-74% still close an `<answer>` tag, and at step 84 the responses
  still quote the prompt's own numbers ("To create an equation using the numbers 26, 23,
  39, and 20 that equals 97, …"); mean length 1400-1600 chars; entropy ~0.6; no NaN. Much
  later the starved policy degrades into tag-repetition babble (steps 150/300: 509/512
  and 506/512 close an `<answer>` tag around mixed-script fragments; still 0 empty, 0
  `!`-only) — drift, not corruption.

Review-side reference arms (NOT in the workspace the diagnosis was made from, and NOT
scoreable as evidence): the 50-step horizon trap (two seeds of this exact config end at
0.584 / 0.514 vs control 0.631 / 0.549 — the collapse is invisible at that budget), the
bf16 300-step control with the identical seed, data, and budget (final 0.678, never below
0.44 after warmup), and the fp8-KV ladder arms (reward 0 from step 1 with prompt-blind
answers — a different, corruption-style family; this run's rollouts quote the prompt's
numbers after the collapse). These live in `reference/ladder_and_control_metrics.txt`,
which only the evaluator sees. A diagnosis that does not cite them must not be marked
false on that account: score `evidence_score` only on the defective-run items above that
the workspace actually contains (the reward trajectory and the rollout-fluency evidence,
either as the oracle's aggregates or as equivalent counts/quotes the diagnosis itself
derives from the dumped files).

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
