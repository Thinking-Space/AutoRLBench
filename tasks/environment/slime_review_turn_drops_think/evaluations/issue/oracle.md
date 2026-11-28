# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the recorded
workspace and the control run that fixed it.

Expected verdict: **unhealthy**.

## Root cause

- `scripts/feedback_rollout.py::generate` (`_render` = `tokenizer.apply_chat_template(messages, ...)`)
  - kind: `environment` / `rollout code`
  - observed:
    - **How the rollout works.** The two-turn answer-then-review rollout keeps an
      OpenAI-style `messages` list and re-renders it with the Qwen3 chat template.
      It does so both to build each turn's generation prompt and, after the last
      turn, to build the training sample (`sample.tokens` = the rendered
      conversation, `loss_mask` = 1 on every `<|im_start|>assistant` span,
      `rollout_log_probs = None`).
    - **What the template does.** The Qwen3 template keeps an assistant turn's
      `<think>...</think>` only for turns after the last user query. For earlier
      assistant turns it keeps just the text after `</think>`; a turn cut off
      before `</think>` is kept verbatim. Turn 1 is followed by the review
      message, so:
      1. the turn-2 prompt contains only turn 1's final summary, not its
         reasoning, and the model reviews a solution it cannot see;
      2. the training sequence is not what the policy generated. Turn 1 is
         trained, with loss mask 1, as `<|im_start|>assistant\n` + final summary
         with no `<think>` (a few hundred tokens instead of the 4–8k the policy
         sampled). GRPO pushes the probability of answering without reasoning
         on tokens the sampler never produced. Only the last turn keeps its
         reasoning.
  - expected: token-in, token-out bookkeeping. Generate each turn from the
    accumulated token ids, append the generated tokens (with log-probs) as
    trainable, and append the user message as non-trainable observation tokens,
    as in slime's `examples/retool/generate_with_retool.py`
    (`sample.append_response_tokens(..., trainable=True/False)`). Alternatively,
    render history with a template that keeps reasoning for every assistant turn.

A passing diagnosis must:

- state that the first answer's `<think>` reasoning is dropped when the
  conversation is re-rendered with the Qwen3 chat template;
- connect this to the training sample: turn 1 is trained as an answer without
  reasoning, on tokens the policy did not generate;
- connect this to the degradation of the review turn.

The turn-2 context losing the reasoning is part of the same mechanism. A
diagnosis that names the context loss and the training-sample mismatch together
is complete. The following are partial or wrong:

- noting only that the review turn got worse;
- noting only that `grad_norm` is high;
- noting only that "the loss mask is wrong", without the template mechanism;
- blaming difficulty, the review prompt, lr or model size.

## Observable impact

- Primary source metric: per-turn correctness in
  `outputs/grpo/rollouts/*.jsonl` (`metadata.turns`) and the final reward
  `rollout/raw_reward` in `outputs/grpo/train.log`.
- Scored evidence (the judge should count these items):
  1. **The review turn degrades.** Joining turn 1 and turn 2 per sample, the
     review changes a correct first answer into a wrong one in 3.2% / 3.9% /
     5.9% of cases (steps 0–9 / 10–19 / 20–29). It fixes a wrong first answer in
     38.5% / 32.4% / 20.1%. Turn-1 accuracy itself stays at 0.67–0.70. Final
     reward drifts 0.789 / 0.748 / 0.720.
  2. **The trained tokens are not the generated tokens.** Each sample's
     `num_tokens` (prompt + trained response) is far below the sum of its
     turns' `gen_tokens`: 61% / 57% / 52% of generated tokens are trained. In
     the shipped responses, 390 of 480 start with the final answer (`To solve…`)
     instead of `<think>`, while the following review turn starts with
     `<think>`. Example: `rollouts/0.jsonl`, the first record: turn 1 generated
     4,610 tokens, but the trained turn 1 is the 840-token summary ending in
     `\boxed{98}`.
  3. **Training symptoms.** `train/grad_norm` is 0.12–0.49 (mean ~0.26) for a
     1.7B model at lr 1e-6. `rollout/log_probs` falls from −0.296 to −0.331 and
     `train/entropy_loss` rises from 0.232 to 0.272. Trained
     `rollout/response_lengths` falls from 6,390 to 4,745, with the turn-1
     generated length down from ~7.4k to ~6.8k.

### Context (not scored)

- `run_grpo.sh`: 30 steps, 32 prompts × 8 samples, temperature 1.0, 12,288
  new tokens per turn, 24,576 context, global batch 128 (two updates per step),
  lr 1e-6 constant, KL coefficient 0, eps clip 0.2 / 0.28, seed 1.
- The generated text of each turn ends with `<|im_end|>`, and the template adds
  another, so rendered turns end `<|im_end|><|im_end|>`. The loss mask stops at
  the first `<|im_end|>`, so the duplicate is not trained. It is a cosmetic
  side effect of the re-render, not the cause.
- Control reference (author-side; not shipped in the workspace and not expected
  in the diagnosis). Same image, seed, data and budget, with a token-in,
  token-out `feedback_rollout.py`:
  - final reward 0.809 / 0.764 / 0.769;
  - review right → wrong 0.3% / 0.4% / 0.4%, wrong → right 39.3% / 33.3% / 30.3%;
  - trained / generated tokens 1.00;
  - `train/grad_norm` 0.05–0.16 (mean ~0.10);
  - `rollout/log_probs` −0.23 to −0.25;
  - turn-1 generated length ~7.4–7.9k.

## Trigger conditions

- A multi-turn RL rollout with a reasoning model whose chat template strips
  `<think>` from earlier assistant turns (Qwen3; also others that drop
  historical reasoning), where the rollout code re-renders the message list
  with `apply_chat_template` to build the next prompt and/or the training
  sample, instead of carrying the generated token ids forward.

## Wrong attributions (must not pass)

- "The policy learns to be concise" or "shorter responses are an efficiency
  gain": the trained sequences are short because the reasoning was cut out of
  them, not because the policy wrote less. Only 52–61% of the generated tokens
  are trained.
- "Self-correction prompts make small models second-guess themselves": the
  control, with the same prompt and model, changes right answers to wrong ones
  in 0.4% of cases, flat over training.
- "MATH Level 5 is too hard" or "1.7B is too small": turn-1 accuracy is
  0.67–0.70 in both arms, and the drift is in the review turn.
- "Learning-rate / batch noise": the flip rates move monotonically over 30
  steps in the recorded run and stay flat in the control.
- "Truncation at 12,288 tokens": truncation falls from 4.7% to 1.3% in the
  recorded run.
- Reward hacking or a reward bug: the grader (`grade_answer_verl` on the
  text after `</think>`) scores the final turn, and the final answers are
  checked against the labels directly.
- "The loss mask covers user tokens": the mask covers assistant spans only.
  The problem is what those spans contain.
- The doubled `<|im_end|>`: it is not trained (see above).

## Documentation reconciliation

- The lower trained length, the high `grad_norm`, the falling log-probs, and
  the degrading review turn are symptoms of the re-rendered conversation, not
  separate causes.
- The Qwen3 chat template and `apply_chat_template` behave as documented;
  dropping reasoning from history is intended for chat inference. The defect is
  using the re-rendered text as both the multi-turn context and the RL training
  sequence.
