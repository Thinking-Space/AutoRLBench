# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

This defect has **two root causes** in the reward path, plus one
configuration factor that removes the last brake. Naming the reward-path cause
with the correct causal role earns partial credit; a complete diagnosis names
both the judge/template cause and the scoring-table cause.

## Root cause 1 of 2

`rl_data/chimera_judge_template_verdict_first.txt` — kind: `reward_hack`

The pairwise judge is shown two candidate answers as `[Response A]` and
`[Response B]` and asked which is "MORE CORRECT". **The template never states
that B is the ground-truth reference.** The judge therefore has no anchor: it
is comparing two pieces of text with no indication that one of them is
authoritatively correct.

```text
You are given one problem and two candidate final answers, labeled A and B.
Decide which one is MORE CORRECT ... If both are equally correct (or equally
wrong), it is a tie.
```

This matters precisely for degenerate answers. Faced with an empty or
content-free answer, a judge that knows B is the reference can say "B is
correct, A is not". A judge that does not know this has no basis to prefer
either, and the template explicitly offers a tie as the answer for
"equally wrong".

- Observed: A/B labels only; no statement that one side is the reference.
  `[[A=B]]` is documented as a valid verdict for "equally correct **or equally
  wrong**".
- Expected: the judge is told which side is the reference (or is asked a
  single-answer equivalence question against a labelled reference), so an
  answer that states no result is scored 0 rather than tied.

## Root cause 2 of 2

`miles/rollout/rm_hub/llm_judge.py::llm_judge_reward` — kind: `reward_hack`

The verdict→score table **pays half credit for a tie**:

```python
LABEL_TO_SCORE = {
    "A>>B": [1] * 3,
    "A>B": [1],
    "A=B": [0.5],      # tie pays 0.5
    ...
}
```

`llm_judge_reward` queries the judge twice with A/B swapped for position
debiasing, then `_compute_reward` averages the two rounds. When the judge
declares a tie in both rounds the sample earns `0.5`. There is no content
check anywhere in this path — unlike `llm_judge_equiv_reward`, which has a
degenerate-answer gate, `llm_judge_reward` forwards whatever the policy
produced, including a bare end-of-sequence token.

- Observed: `"A=B": [0.5]` combined with a judge that cannot distinguish a
  content-free answer from a wrong one.
- Expected: a degenerate or empty answer scores 0 deterministically, and a tie
  between a real attempt and a content-free answer is not reachable.

## Configuration factor (not a root cause on its own)

The launch line passes **no** `--kl-loss-coef`, so the KL term is at its
default of 0. With no anchor to the initial policy, once the tie payout makes
short answers profitable the policy can move all the way to a one-token
output. Stating "KL is zero" as *the* root cause is incomplete: without the
tie payout there is no gradient pushing toward empty answers in the first
place.

## Causal chain

The template gives the judge no way to identify the reference; the judge
therefore calls an unreadable answer a tie; the scoring table pays the tie
0.5; GRPO observes that a one-token completion out-earns a full chain of
thought that happens to be wrong; the policy collapses to that one token.

Measured payout inversion:

| phase | mean response length | mean raw reward |
| --- | ---: | ---: |
| healthy, steps 0–10 | 5842.7 | 0.1731 |
| transition, steps 11–22 | 2257.6 | 0.2456 |
| early collapse, steps 23–40 | 25.1 | 0.3004 |
| full collapse, steps 41–137 | 1.2 | 0.3214 |

A one-token answer earns **1.86×** what a genuine 5.8k-token attempt earned
before the collapse.

## Observable impact

- Primary metrics: `rollout/response_lengths` against `rollout/raw_reward`,
  both in the frozen per-step trace.
- The run covers rollout steps 0 through 137 and was stopped by the operator.
- `rollout/response_lengths` falls from 5801 tokens at step 0 to exactly 1.0
  token, reached by step 134 and held to the end. Mean length is still 297
  tokens at step 22 and already 12.9 at step 23, so the collapse is a narrow
  band, not a gradual drift.
- `rollout/raw_reward` **rises** across the same window, from 0.175 to 0.365
  (peak 0.411), so the reward curve read on its own says the run is improving.
- 103 of 138 steps produce ≤2-token completions at a mean reward of 0.321 —
  higher than any phase with real generation.
- Held-out accuracy is effectively zero: a one-token completion cannot contain
  a STEM answer.
- Frozen evidence is in `outputs/metrics.json` and
  `outputs/historical_train_trace.log`.
- Rollout dumps for this job were deleted before packaging, so the literal
  one-token outputs are not shipped; the collapse is evidenced by the per-step
  trace.

## Trigger conditions

- `--rm-type llm_judge`, so scoring goes through the pairwise path and the
  tie-pays-0.5 table.
- `--judge-prompt-template` points at the A/B template that never identifies
  the reference, and `--judge-kwargs enable_thinking=false` leaves the judge
  no reasoning budget.
- No `--kl-loss-coef`, so no anchor prevents full drift.

## Documentation reconciliation

- **"Training did not improve" is the wrong diagnosis.** `rollout/raw_reward`
  genuinely rises. The failure is that the rising reward tracks a degenerate
  policy.
- Naming only `kl-loss-coef` being absent is incomplete: it is a brake, not the
  engine. The engine is the tie payout made reachable by the un-anchored judge.
- `enable_thinking=false` is present in the control arm as well and is not by
  itself the defect; it removes the judge's chance to reason about whether an
  empty answer should count.
- Truncation is unrelated: `--zero-reward-on-truncation` is not set, and the
  truncated ratio falls to zero as answers shorten, which is the opposite of a
  truncation problem.
- The judge model and endpoint are not the root cause: a judge of the same
  family scores correctly once it is told which side is the reference.
- The dual EOS ids `[1, 130073]`, `rope_theta 5e6`, the DSpark speculative
  decoding block, R3 routing replay, TIS/icepop, and the CP/EP parallel layout
  are expected properties of this recipe, not defects.
- A falling `rollout/response_lengths` is the symptom. The root cause must be
  stated in terms of the template and the scoring table, which are
  configuration and code, not in terms of the curve itself.
