# Canonical issue oracle

Reconciled from the buggy/baseline source comparison and the frozen run logs.

Expected verdict: **unhealthy**.

## Root cause

`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` — kind: `symbol`

- **Observed (workspace):** while building the long-lived vLLM `SamplingParams`, the generic config loop accepts every config key for which `hasattr(SamplingParams(), str(k))` is true. Because `seed` is a `SamplingParams` field in this runtime, `actor_rollout_ref.rollout.seed=123` is copied into that shared sampling-parameter object.
- **Expected (baseline):** the loop must not forward `seed` into this shared `SamplingParams` object; the baseline condition explicitly excludes `k == "seed"`.
- **Causal role:** retaining the same explicit seed on the shared sampling parameters resets sampling randomness across rollout generation calls. This undermines diverse group samples used by GRPO and can leave policy updates with zero/near-zero signal. The log pattern is consistent with this: the observed run reports `actor/grad_norm:0.0` and `actor/pg_loss:0.0` at all 50 steps, while the baseline comparison has nonzero gradient norms on some steps.

## Observable impact

- The observed run reports `val-core/openai/gsm8k/reward/mean@1:0.734375` at each of its ten evaluation points (steps 5–50).
- The same held-out first-256 GSM8K examples scored `0.73828125` before training with the baseline workspace.
- The baseline run changes across its ten evaluation points, from `0.78125` at step 5 to `0.8828125` at step 50.
- These are observed single runs, not estimates of run-to-run variance. The metric curves are symptoms/evidence; the source-level seed forwarding is the root cause.

## Trigger conditions

- `actor_rollout_ref.rollout.seed` is set to `123` by the launch script.
- The installed vLLM `SamplingParams` exposes a `seed` attribute, so the generic attribute check accepts the key.
- GRPO uses multiple rollouts (`n=8`) per input, so reduced sampling diversity can reduce within-group learning signal.

## Scope notes

- The only substantive difference between the two source snapshots is the added `and k != "seed"` condition on this line. Their training launch scripts are identical.
- `data.max_response_length=4096`, `rollout.n=8`, the GSM8K preprocessing, and the single-GPU setting are shared by both arms; do not attribute the observed A/B difference to these shared settings without additional evidence.
- The constant validation score is a symptom, not the code-level cause.
