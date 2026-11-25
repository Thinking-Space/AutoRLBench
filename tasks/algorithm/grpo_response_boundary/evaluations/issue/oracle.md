# Canonical diagnosis

Expected verdict: **unhealthy**.

## Root cause

`workspace/vendor/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`, `vLLMRollout.generate_sequences`: in regular sampled rollout, the padded response mask comes from `get_response_mask` using the checkpoint's EOS ID. The native Nemotron checkpoint publishes EOS 2, while its chat tokenizer ends the generated responses with ID 11 and the task uses this same ID for response padding. Those completions actually end. After padding the responses, the EOS 2 scan cannot recognize the true boundary, and all 16,384 positions can be marked valid.

The returned sequence length before padding is the authoritative response boundary. Mask positions at or beyond that length. This retains a returned stop token and handles a stop token omitted by the engine. Merely scanning for the pad ID or adding chat EOS to the old scan misses cases because EOS 11 can be a real emitted token or padding, and a stop may not be returned.

## Observable impact

- In `workspace/experiment_logs/train.log`, both GRPO steps report 16,384 for the response length minimum, mean, and maximum and a clip ratio of 1.0. The training rollouts preserve complete decoded answers whose lengths visibly differ. In the full run records, 25/32 and 20/32 decoded responses re-encode below the budget at steps 1 and 2. In `workspace/generation_records/engine-requests.json`, the two original engine outputs contain 2,911 and 13,430 tokens and both stopped normally.
- The naive reward manager decodes with special tokens skipped and can still produce correct binary outcome rewards. Its nonzero rewards land at the last position selected by the incorrect mask. GRPO broadcasts the centered outcome advantage across that same mask, and the native clipped token-mean objective consumes those positions.
- The relevant evidence is the disagreement for short, normally stopped completions and the consequence for nonzero reward placement and token credit. Numerical convergence or held-out improvement is not required to diagnose it.

## Documentation reconciliation

Long generation, evaluation noise, prompt left padding, and incorrect boxing do not account for right padding becoming valid after a normally stopped completion. The scorer and GRPO are consuming a faulty upstream mask.
