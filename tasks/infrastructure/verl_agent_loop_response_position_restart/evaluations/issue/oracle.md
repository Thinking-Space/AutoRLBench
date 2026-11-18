# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, data, seed and budget; only
`verl/experimental/agent_loop/agent_loop.py` differs). That difference is the task-injected
defect: upstream verl 0.8.0 numbers the concatenated window once, and the recorded control's
unchanged file matches upstream (see `environment/source_manifest.json`).

Expected verdict: **unhealthy**.

## Root cause

- `verl/experimental/agent_loop/agent_loop.py::AgentLoopWorker._compute_position_ids` (text-only
  branch)
  - kind: `symbol`
  - observed: the prompt window (left-padded to `prompt_length`) and the response window
    (right-padded) are numbered separately, so response token k gets position id k, the same
    positions as the first prompt tokens, instead of continuing after the prompt.
    vLLM computes its own positions from the token sequence, so rollouts and validation outputs
    come from the real policy; the actor (old log-probs and the update) and the reference model
    read `position_ids` from the batch and score every rollout with the response placed on top of
    the prompt in RoPE position space.
  - expected: one cumulative count over the concatenated mask (`compute_position_id_with_mask`),
    as in the control.

Naming the separate numbering of the response window with its causal role (the actor and
reference score rollouts under position ids that differ from the ones the samples were generated
with → the policy gradient is computed for a different conditional distribution than the one vLLM
samples → updates transfer poorly to the sampled policy, the actor drifts far in its own view
while held-out accuracy barely moves) is a complete attribution. Do not require additional causes.

**The position-id defect is required.** "Train/inference mismatch", "the actor's log-probs are
off" or "KL grows too fast" without locating the response position ids earns at most partial
root-cause credit. Blaming the format loophole (the prompt asks for `####`, the flexible scorer
takes the last number, the policy moves to `\boxed{}`; the control has the same loophole and
reaches 0.5580), sampled-versus-greedy sharpening, overfitting, the learning rate, the KL
coefficient, the 500-character data filter, or calling the run healthy is wrong and gets no
root-cause credit.

## Observable impact

Every item here is derivable from the shipped workspace (`results/run60` plus the code and
configuration).

- Primary source evaluation metric: logged `val-core/openai/gsm8k/acc/mean@1` 0.433 (0) → 0.437
  (10) → 0.450 (20) → 0.465 (30) → 0.488 (40) → 0.465 (50) → 0.477 (60). Re-scoring
  `val_generations/<step>.jsonl` gives 0.4632, 0.4602, 0.4564, 0.4670, 0.4875, 0.4678, 0.4776: a
  small gain that stalls after step 40.
- Training reward (`critic/score/mean`) dips and then rises: 0.441 (step 1), 0.375 (10), 0.484
  (20), 0.482 (30), 0.484 (40), 0.609 (50), 0.635 (60), so the train curve and held-out accuracy
  separate.
- The actor moves far from the reference in its own view: `actor/kl_loss` 0.0004 (1), 0.014 (10),
  0.065 (20), 0.113 (30), 0.196 (40), 0.322 (50), 0.441 (60) with `kl_loss_coef=0.001`;
  `actor/grad_norm` 1.91 at step 1 and rising to 2.1-5.1 from step 40 on.
- `actor/entropy` (computed by the actor under the batch's position ids) starts at 0.82 and falls
  to 0.34.

## Trigger conditions

- A position-id construction that is used by the trainer but not by the inference engine: the
  rollout engine derives positions from the token sequence itself, so nothing in generation or
  reward reveals the mismatch.
- The importance ratio and the KL are both computed in the same (wrong) view, so `ppo_kl`, the
  clip fraction and the KL term look self-consistent.

## Documentation reconciliation

- The code comment claims the two windows must be numbered separately because they are padded on
  different sides; `compute_position_id_with_mask` over the concatenated mask already skips the
  left padding and continues through the response. The high KL, the rising gradient norm, the
  entropy level and the stalled held-out accuracy are symptoms of the same defect.
