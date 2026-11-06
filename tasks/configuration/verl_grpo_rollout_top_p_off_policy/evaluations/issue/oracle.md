# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, data, seed and budget; only `ROLLOUT_TOP_P` in `run_gsm8k_grpo.sh`
differs: 0.5 here, 1.0 in the control).

Expected verdict: **unhealthy**.

## Root cause

- `run_gsm8k_grpo.sh` (`ROLLOUT_TOP_P=0.5` → `actor_rollout_ref.rollout.top_p=0.5`)
  - kind: `config`
  - observed: training rollouts are sampled by vLLM from the top-50% nucleus at temperature 1.0,
    but the update treats them as samples of the full policy: `old_log_probs` are recomputed by the
    FSDP actor over the full softmax and `algorithm.rollout_correction` is disabled
    (`rollout_is: null`, `bypass_mode: false`), so the vLLM log-probs only feed the `rollout_corr/*`
    metrics. Tokens outside the nucleus are never sampled and never receive a negative advantage,
    while negative-advantage updates lower the logits of the sampled tokens and shift that mass into
    the unsampled tail. The full-distribution entropy grows, and once the top token of a position
    drops below 0.5 the inflated tail enters the nucleus and the response derails into random
    vocabulary. Validation is greedy (`val_kwargs.do_sample=False`), so it is hit later and less, but
    it still degrades.
  - expected: training rollouts sampled from the policy the update assumes, `top_p=1.0` (verl's
    default and every verl GRPO example), or an explicit correction for the truncated sampler.

Naming this mismatch with its causal role (nucleus-truncated samples trained as on-policy → unsampled
tail never penalized → entropy explosion and garbage continuations → collapsed group diversity,
truncation and a held-out peak at step 10 followed by decline) is a complete attribution. Do not
require additional causes.

**The setting and the mechanism are required.** "Entropy exploded", "the policy drifted from the
reference" or "rollouts degenerated" without tracing it to `top_p=0.5` used by an uncorrected,
full-softmax update earns at most partial root-cause credit. Naming `top_p=0.5` only as one of several
co-equal causes next to a weak KL coefficient is also partial. Calling the run healthy because
held-out accuracy rose from 0 to about 0.49, or blaming the KL coefficient, the 512-token response
cap, the learning rate, the entropy coefficient, loss aggregation, the reward function or the data,
is wrong and gets no root-cause credit.

## Observable impact

The evidence items are the six numbered facts below; score each as one boolean (values may be
approximate, the direction must be correct). Every item is derivable from the shipped workspace
(`results/run100` plus the code and configuration).

1. The sampler differs from the trained policy before any update: at step 1 (zero gradient, all
   1,024 rollouts scored 0) `rollout_corr/kl` is 0.086, `training/rollout_probs_diff_mean` 0.065
   and `training/rollout_actor_probs_pearson_corr` 0.71 — any one of the three stated with its
   direction counts as the item.
2. `actor/entropy` explodes: 0.26 (step 1) → 2.25 (17), never below 1.5 afterwards, peak 5.06
   (63), still 2.86 (100).
3. Responses derail into truncation and drift: `response_length/clip_ratio` 0.04 (steps 1 to 13)
   → 0.31 (17), peak 0.37 (63), 0.18 (100); `actor/kl_loss` 0.035 (10) → 0.52 (63) with
   `kl_loss_coef=0.001` — either metric stated with its direction counts as the item.
4. Response content degenerates: in `rollouts/{1,50,100}.jsonl`, mixed-script garbage (CJK,
   Cyrillic, Arabic or Hangul runs inside an English solution) in 0 / 227 / 188 of 1,024 responses,
   all scored 0; groups of eight byte-identical responses 0 / 32 / 45 of 128; zero-advantage groups
   128 / 111 / 109; greedy held-out answers with a repetition loop (a 2 to 40 character unit
   repeated 12 or more times) 1 at step 0, 200 at step 50, 175 at step 100 — reporting the
   degeneration with at least one of these families quantified counts as the item.
5. Held-out accuracy peaks at the first evaluation after the format is learned and then declines:
   `val-core/openai/gsm8k/acc/mean@1` 0.001 (0) → 0.500 (10) → 0.446 (30) → 0.486 (100).
6. The run still learns: training score `critic/score/mean` 0.000 (1) → 0.62 (10), then 0.45 to
   0.67 with no trend through step 100 (0.58).

## Trigger conditions

- Training rollouts sampled with `top_p < 1` while the policy-gradient update recomputes the
  behavior probabilities from the full softmax and applies no importance or rejection correction.
- A low KL coefficient (verl's 0.001 here, shared with the control) does not anchor the unsampled
  tail, because the KL term is evaluated only at sampled tokens.

## Documentation reconciliation

- Control reference (author-side; same code, data, seed and budget with `ROLLOUT_TOP_P=1.0`; not
  shipped in the workspace, rebuttal context only): held-out strict accuracy 0.001 → 0.498 (10) →
  0.542 (20) → 0.569 (50) → 0.566 (100); entropy 0.55 → 0.06; clip ratio 0.11 → 0.02; `kl_loss`
  0.119 at step 100; `rollout_corr/kl` 0.0005; garbage in 5 / 4 / 10 rollouts; no identical groups;
  zero-advantage groups 113 / 62 / 70; no repetition loops at step 100.
- "KL too weak" is wrong as a root cause: the coefficient is the same in the control, where the
  policy stays close to the reference.
- "The 512-token cap is too tight" is wrong: the control's clip ratio is 2%; the cap is hit by
  garbage continuations, not by long solutions.
- "The run is healthy" is wrong: held-out accuracy peaks at step 10 and ends 0.08 below the control,
  with a degeneration mode the base model does not have.
- The reward function, the DAPO reward manager (overlong penalty disabled), the preprocessing, GRPO
  advantage, learning rate, entropy coefficient (0) and loss aggregation are unchanged from the
  control.
- "Training did not improve" is wrong: held-out strict accuracy rises from 0.001 to 0.500 by step 10
  and the training score to about 0.6. The defect makes the run end below the control (0.486 against
  0.566) and degrade after step 10; it does not stop learning.
- The ray/vLLM warnings, `trainer.use_v1=False`, W&B offline mode and the wandb teardown traceback
  at exit are environment noise.
