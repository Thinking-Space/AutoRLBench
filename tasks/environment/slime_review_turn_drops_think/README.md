# Re-rendering a multi-turn conversation with the Qwen3 chat template drops the first answer's reasoning from GRPO training

GRPO on MATH Level 5 with Qwen/Qwen3-1.7B (thinking mode) and slime, in a two-turn answer-then-review environment: the policy solves the problem, the user asks it to review its solution, and the reward is 1 if the final answer is correct. The custom rollout builds both the next-turn prompt and the training sample by re-rendering the message list with `tokenizer.apply_chat_template`. The Qwen3 chat template removes the `<think>...</think>` block of every assistant turn that comes before the last user message. So the review turn never sees the first answer's reasoning, and GRPO trains the first turn as a bare answer that the policy never produced. Nothing errors. Reward stays near 0.7–0.8, and the first turn even looks more "efficient". But the review turn steadily loses the ability to fix mistakes and starts breaking correct answers.

| | |
| --- | --- |
| **Author** | Sikuan Yan |
| **Affiliation** | LMU Munich |
| **Type** | Environment |
| **Domain** | Math |
| **Tags** | `chat-template` `multi-turn` `re-tokenization` `off-policy-tokens` `self-correction` |
| **Framework** | slime v0.3.2 (image `/root/slime`, commit `a3f5009`, copied into `workspace/slime` for reading), Megatron backend, SGLang rollout, GRPO, custom generate function `scripts/feedback_rollout.py` |
| **Run time** | ~4 hours (30 GRPO steps of ~8 minutes, after a ~2-minute checkpoint conversion) |
| **Resources** | 2 × H100 (80 GB) |

## Difficulty

`run_grpo.sh` is an ordinary slime GRPO recipe: 32 prompts × 8 samples per step, temperature 1.0, 12,288 new tokens per turn, lr 1e-6, no KL, 30 steps. Actor and SGLang are colocated on two GPUs. The environment lives in `scripts/feedback_rollout.py`, plugged in with `--custom-generate-function-path` and `--custom-rm-path`. It keeps an OpenAI-style `messages` list. Before each turn it renders the list with `apply_chat_template(..., add_generation_prompt=True)` and generates from those tokens. After the last turn it renders the whole conversation again. It uses the rendered tokens as `sample.tokens` and sets a loss mask of 1 on every `<|im_start|>assistant` span. It does not keep rollout log-probs, so slime recomputes them.

The Qwen3 chat template drops reasoning from history on purpose. For an assistant message that comes before the last user query, it keeps only the text after `</think>`. A turn cut off before `</think>` is kept verbatim. Inference servers use the same rule, so the template behaves exactly as documented. In this environment, though, the rule has two effects:

- **Generation.** The turn-2 prompt contains the first answer's final summary but none of its reasoning, often thousands of tokens missing. The model is asked to "review your solution" without seeing it.
- **Training.** The training sample is the re-rendered conversation, not the token stream the policy generated. Turn 1 appears as `<|im_start|>assistant\n` followed directly by the final summary. In the recorded run that is a few hundred tokens where the policy actually generated 4–8k, and it starts without `<think>`. GRPO then pushes up the probability of answering without reasoning, on tokens the sampler never produced. Turn 2, the last turn, keeps its `<think>` block.

The symptom is far from the cause. The run looks healthy: reward is 0.72–0.79 per 10-step window, `pg_clipfrac` is about 3e-4, nothing is truncated badly, and there are no warnings. Trained response length falls from 6,390 to 4,745 tokens and truncation from 4.7% to 1.3%, which reads as the policy becoming more efficient. The reward drift (0.789 → 0.720) is small next to batch-to-batch noise. The decisive evidence is per turn and per token:

- `outputs/grpo/rollouts/*.jsonl` record each turn's `gen_tokens`, `correct`, `finish` and `pred` in `metadata.turns`. Comparing turn 1 with turn 2 shows the review turn changing right answers to wrong ones more and more often (3.2% → 5.9%) and fixing wrong ones less and less often (38.5% → 20.1%).
- Each sample's `num_tokens` is far below the sum of its turns' `gen_tokens`. Only 52–61% of the generated tokens are in the trained sequence.
- In the shipped responses, 390 of 480 begin `To solve…` rather than `<think>`. The prompt ends at `<|im_start|>assistant\n`, so the trained turn 1 opens directly with the answer.
- `train/grad_norm` is 0.12–0.49, against 0.05–0.16 in the control, and `rollout/log_probs` falls to −0.33 against −0.24: the trained tokens are unlikely under the policy.

Plausible but wrong readings:

- "The model is learning to be concise / reward hacking by shortening."
- "Review prompts make small models second-guess correct answers; that is a known weakness of self-correction."
- "lr or batch noise."
- "MATH Level 5 is too hard for 1.7B."
- "Truncation at 12k."

| per 10-step window (steps 0–9 / 10–19 / 20–29) | recorded (re-rendered) | control (token-in, token-out) |
| --- | ---: | ---: |
| final reward (`rollout/raw_reward`) | 0.789 / 0.748 / 0.720 | 0.809 / 0.764 / 0.769 |
| turn-1 accuracy | 0.693 / 0.667 / 0.701 | 0.689 / 0.650 / 0.672 |
| review: right → wrong | 3.2% / 3.9% / 5.9% | 0.3% / 0.4% / 0.4% |
| review: wrong → right | 38.5% / 32.4% / 20.1% | 39.3% / 33.3% / 30.3% |
| turn-1 generated tokens | 7,406 / 7,614 / 6,780 | 7,458 / 7,868 / 7,390 |
| trained / generated tokens | 0.61 / 0.57 / 0.52 | 1.00 |
| trained response length (`rollout/response_lengths`) | 6,390 / 5,994 / 4,745 | 9,478 / 9,939 / 9,296 |
| `rollout/log_probs` | −0.296 / −0.302 / −0.331 | −0.233 / −0.241 / −0.245 |
| `train/entropy_loss` | 0.232 / 0.247 / 0.272 | 0.232 / 0.241 / 0.244 |
| `train/grad_norm` (mean per window of 20 updates) | 0.242 / 0.266 / 0.268 | 0.088 / 0.111 / 0.097 |

**Horizon.** One GRPO run of 30 steps (rollout batch 32 × 8, global batch 128 = two updates per step). There is no in-run evaluation. No held-out set is used: every rollout sample is graded per turn (`metadata.turns[*].correct` against the MATH reference answer, independent of the reward the policy was trained on), and these per-turn records are the independent evidence. The frozen evidence is:

- the training log with effective arguments and per-step `rollout N` and `step N` metrics (`outputs/grpo/train.log`);
- the per-sample rollout records of all 30 steps (`outputs/grpo/rollouts/{0..29}.jsonl`, converted from slime's `--save-debug-rollout-data` dumps; full response text for the first two prompt groups of each step);
- the data and its build script;
- slime's source.

**Recorded and control runs.** Both arms ran once (seed 1, same prompt order) in the shared slime v0.3.2 image on 2 × H100, with start model `Qwen/Qwen3-1.7B` (revision `70d244c`). The control differs only in `scripts/feedback_rollout.py`. It keeps the token stream like slime's own multi-turn `examples/retool` generate function: it appends each turn's generated tokens with their log-probs as trainable, and appends the review message as non-trainable observation tokens. So turn 2 is generated after the full first answer, and the trained tokens are exactly the sampled ones.

**Turn-1 accuracy is not the signal.** Turn-1 accuracy stays similar in both arms. 30 steps at lr 1e-6 are not enough to remove reasoning from turn 1. The damage concentrates where both effects meet: the review turn works from an answer with no visible derivation, and training rewards the policy for answering without reasoning.

**Origin.** The defect was found while building a multi-turn self-correction environment for an automated post-training experiment. A message-list re-render is the natural first implementation; slime's `retool` example and other agentic RL frameworks avoid it with token-in, token-out bookkeeping.

**Startup rerun.** The official rerun, effectively a second run of the recorded side, reproduced the drift (reward 0.779 / 0.734 / 0.726; review right → wrong 3.5% / 5.1% / 5.1%, wrong → right 36.8% / 25.5% / 22.0%), with one `train/grad_norm` spike of 65.9 at update 23 (clipped at `clip_grad` 1.0) that the recorded run does not have.

**Data.** `data/math_l5_train.jsonl` holds the 2,264 Level 5 problems of the MATH train split whose last `\boxed{}` answer is at most 40 characters. They come from `DigitalLearningGmbH/MATH-lighteval` (MIT, revision `0530c78`, shipped as `data/math_lighteval_train.parquet`) and were built with `scripts/make_data.py`. Grading uses slime's `grade_answer_verl` on the text after `</think>` (or the whole turn if it stopped without closing the block).

**Issue-only.** Each arm ran once. Final reward differs by a few points, within what one seed can show. The decisive signals are the reasoning missing from both the turn-2 context and the training tokens, and the per-turn flip rates. These call for a diagnosis rather than a rerun threshold.

## Reference solution

- **issue**: identify that `scripts/feedback_rollout.py` re-renders the message list with the Qwen3 chat template. Because the template strips `<think>...</think>` from assistant turns before the last user message:
  - the turn-2 (review) prompt lacks the first answer's reasoning;
  - the training sample is not the generated token stream: turn 1 is trained as a bare final answer with loss mask 1, on tokens the policy never sampled.

  The fix is token-in, token-out bookkeeping: accumulate generated tokens and log-probs per turn, and append the user turn as non-trainable tokens.

  Supporting evidence:
  - turn-level flips in `metadata.turns`: right → wrong rises to 5.9%, wrong → right falls to 20.1%;
  - `num_tokens` far below the sum of `gen_tokens`;
  - trained responses that begin without `<think>`;
  - high `grad_norm` and low `rollout/log_probs`.

  Attributions to model size, task difficulty, the review prompt, lr, truncation or reward hacking do not pass. Noting only that the review turn got worse, without the template mechanism, is partial.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict, root cause and evidence, weighted 0.2 / 0.6 / 0.2. As implemented in the shared `judge.py`, passing requires a verdict score of 1.0 and root-cause and evidence scores above 0.8.
