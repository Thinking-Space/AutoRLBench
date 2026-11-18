# Helper and chat-template reproduction

This note is reviewer evidence, not an agent-visible file. It makes the root cause reproducible from
the shipped code and chat template. The original rollout dumps were deleted before packaging, so this
is a constructed reproduction of the helper's behavior, not a frozen rollout sample; the frozen
per-step trace under `workspace/outputs/` is the run-level evidence.

## The buggy helper

`workspace/miles/rollout/rm_hub/llm_judge.py::extract_non_reasoning_content`:

```python
def extract_non_reasoning_content(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from text."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
```

## The chat template prefill

`workspace/model/chat_template.jinja`, `add_generation_prompt` block:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined %}
        {%- if enable_thinking is false %}
            {{- '<think>\n\n</think>\n\n' }}
        {%- elif enable_thinking is true %}
            {{- '<think>\n' }}
        {%- endif %}
    {%- endif %}
{%- endif %}
```

With `enable_thinking=true` (the recorded launch command passes
`--apply-chat-template-kwargs '{"enable_thinking": true}'`), the opening `<think>` is written into the
prompt. The model response therefore has the shape `…reasoning…</think>answer`: it carries only the
closing tag.

Note the asymmetry: with `enable_thinking=false` the template emits both tags into the prompt, and the
paired regex would work. The defect only bites in the thinking-enabled configuration the recorded run
used.

## Reproduction

```text
$ python3 -c 'import re; f=lambda t: re.sub(r"<think>[\s\S]*?</think>", "", t).strip();
print(repr(f("reasoning...</think>final answer")))'
'reasoning...</think>final answer'
```

The input comes back unchanged, so the judge receives the reasoning block and the answer together.
A correct implementation cuts at the last closing tag:

```python
"reasoning...</think>final answer".split("</think>")[-1].strip()
# 'final answer'
```

The shipped test file `workspace/tests/fast/rollout/rm_hub/test_llm_judge.py` only asserts paired-tag
inputs such as `<think>reasoning</think>The answer is 42.`, which the buggy helper passes; it does not
cover the closing-tag-only shape produced by this chat template.
