# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

`miles/rollout/rm_hub/llm_judge.py::extract_non_reasoning_content` — kind:
`reward_hack`

The judge is supposed to see only the final answer, never the reasoning. The
helper that removes reasoning assumes the *response* contains a matching pair of
tags:

```python
def extract_non_reasoning_content(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from text."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
```

But the chat template prefills the opening `<think>\n` at the **end of the
prompt**. The decoded response is therefore not
`<think>…</think>answer` but `…</think>answer`: the opening tag was never
generated, only the closing one was. A paired regex finds no match, so it
removes nothing and hands the entire reasoning block, plus the answer, to the
judge.

- Observed: `re.sub(r"<think>[\s\S]*?</think>", "", …)` applied to a response
  that has a closing tag only; the input comes back unchanged.
- Expected: reasoning is stripped regardless of which side supplied the opening
  tag — e.g. cut at the last `</think>` when no opening tag is present.

The template itself is shipped at `model/chat_template.jinja`; its
`add_generation_prompt` block makes the prefill explicit:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- elif enable_thinking is true %}
        {{- '<think>\n' }}          <-- opening tag goes into the prompt
    {%- endif %}
{%- endif %}
```

Note the asymmetry: with `enable_thinking=false` the template emits *both* tags
into the prompt, and the paired regex would then work. The defect only bites in
the thinking-enabled configuration the recorded run used.

The launch line makes reasoning mode explicit
(`--apply-chat-template-kwargs '{"enable_thinking": true}'`), so the responses
are long and the whole block reaches the judge on every sample. A runnable
reproduction of the helper and the template prefill is in
`evaluations/issue/reference/evidence.md`.

## Observable impact

- Every judge call carries the reasoning block. In the recorded run the judge's
  response input had a median of about 21,485 characters, while the final
  answer after the closing tag is a few hundred; stripping correctly drops the
  median to about 1,164 characters (about 18×). These figures come from the
  original run's rollout dumps, which were deleted before packaging; the
  mechanism above is reproducible from the shipped code alone.
- A judge that must locate the final answer inside thousands of tokens of
  reasoning is noisy: intermediate, later-abandoned conclusions compete with the
  answer it is asked to grade.
- The frozen trace is entirely normal — `rollout/raw_reward` rises from 0.324
  to 0.617 across 717 steps, response length stays around 8k tokens, nothing
  errors. There is no metric that reports what the judge actually received.
- Frozen evidence is in `outputs/metrics.json` and
  `outputs/historical_train_trace.log`.

## Trigger conditions

- `--rm-type llm_judge_equiv`, so scoring goes through this helper.
- `--apply-chat-template-kwargs '{"enable_thinking": true}'`, so responses are
  `reasoning</think>answer` and never contain the opening tag.

## Documentation reconciliation

- **"Training did not learn" is the wrong diagnosis.** The reward curve rises.
- Naming only "the regex is wrong" is partial: the reason it never matches is
  the chat template's prefilled opening tag, and a fix that only loosens the
  regex without accounting for the prefill will strip the wrong region.
- Blaming the judge model is wrong: the same judge grades correctly once it is
  given only the answer.
- Blaming response length or truncation is wrong: long responses are the
  intended effect of thinking mode, and the truncated fraction is small.
- `kl-loss-coef 0.001`, the judge kwargs (`temperature=0.0`, `max_tokens=4096`)
  and the reason-first judge template are expected settings of this recipe.
- A single root cause is present; a diagnosis that names the function and the
  prefilled-tag mechanism is complete, and one that names only one half earns
  partial credit.
