# Canonical diagnosis

Expected verdict: **unhealthy**.

## Root cause

`workspace/vendor/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`, `vLLMRollout.generate_sequences`: in regular sampled rollout, the padded response mask comes from `get_response_mask` using the checkpoint's EOS ID. The native Nemotron checkpoint publishes EOS 2, while its chat tokenizer ends the generated responses with ID 11 and the task uses this same ID for response padding. Those completions actually end. After padding the responses, the EOS 2 scan cannot recognize the true boundary, and all 16,384 positions can be marked valid.

The returned sequence length before padding is the authoritative response boundary. Mask positions at or beyond that length. This retains a returned stop token and handles a stop token omitted by the engine. Merely scanning for the pad ID or adding chat EOS to the old scan misses cases because EOS 11 can be a real emitted token or padding, and a stop may not be returned.

## Observable impact

Each bullet below is one evidence item. Where a bullet names two aspects of one consequence, a diagnosis that states either aspect with the correct direction has stated that item.

- In `workspace/experiment_logs/train.log`, both GRPO steps report 16,384 for the response length minimum, mean, and maximum and a clip ratio of 1.0.
- `critic/score/mean` runs 0.4375 and 0.375 (14/32 and 12/32 correct answers), so the outcome rewards are right.
- The training rollouts preserve complete decoded answers whose lengths visibly differ and fall far short of the 16,384-token budget — the uniform logged lengths disagree with the shipped records.
- In `workspace/generation_records/engine-requests.json`, the two original engine outputs contain 2,911 and 13,430 tokens and both stopped normally.
- The naive reward manager decodes with special tokens skipped and can still produce correct binary outcome rewards.
- The placement of each nonzero reward follows the incorrect mask — the reward is written at the mask's end-of-response position, its last selected position.
- GRPO broadcasts the centered outcome advantage across that same mask, and the native clipped token-mean objective consumes those positions.

The relevant evidence is the disagreement for short, normally stopped completions and the consequence for nonzero reward placement and token credit. Numerical convergence or held-out improvement is not required to diagnose it.

## Trigger conditions

- A rollout whose response mask is derived by scanning for a fixed end-of-sequence token ID.
- The served model's chat template ends completions with a different token ID than the checkpoint's published EOS, and the same ID doubles as the padding fill.

## Documentation reconciliation

Long generation, evaluation noise, prompt left padding, and incorrect boxing do not account for right padding becoming valid after a normally stopped completion. The scorer and GRPO are consuming a faulty upstream mask.

Author-side context, not required agent evidence: in the full 32-answer run records, 25/32 and 20/32 decoded responses re-encode below the budget at steps 1 and 2; a controlled replay of the 2,911-token answer keeps its score at 1 under both masks while moving the reward from position 16,384 to position 2,911, and the length-based mask reduces padding positions carrying nonzero gradient with respect to token log probability for a mixed pair from 6,521 to 0.
