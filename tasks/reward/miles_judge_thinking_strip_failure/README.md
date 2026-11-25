# Judge reads the whole reasoning block

Training MiniCPM5-2.6B with GRPO on a CHIMERA pass@1-6 set (SFT base trained on GPT-OSS-synthesized data); the helper that is supposed to strip the model's reasoning before judging removes nothing, because the chat template prefills the opening `<think>` into the prompt, so the judge grades a multi-thousand-token reasoning block on every sample while every logged metric looks normal.

| | |
| --- | --- |
| **Author** | Bingxu An |
| **Affiliation** | University of Chinese Academy of Sciences |
| **Type** | Reward |
| **Domain** | STEM |
| **Tags** | `llm-judge` `thinking-mode` `regex` `silent-regression` |
| **Framework** | miles (slime fork); GRPO; judge `deepseek-v4-flash` |
| **Run time** | 717 rollout steps (operator-stopped; wall-clock duration not recorded) |
| **Resources** | 16 × H800 (80 GB), two nodes |

## Difficulty

Recorded runs: defective **619439** (717 rollout steps). The control recipe —
the same task with a judge that receives only the answer — is **630255**
(595 steps).

The defect is a four-line helper that reads correctly in isolation:

```python
def extract_non_reasoning_content(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from text."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
```

The regex needs **both** tags. The chat template, however, prefills the opening
`<think>\n` at the end of the *prompt*: the model never generates it, so the
decoded response is `…reasoning…</think>answer`. The pair never appears, the
substitution matches nothing, and the judge is handed the reasoning block and
the answer together.

The template that does the prefill ships in the workspace at
`model/chat_template.jinja`. Its `add_generation_prompt` block emits
`<|im_start|>assistant\n` and then, when `enable_thinking` is true,
`<think>\n` — the opening tag is written into the prompt, never generated. With
`enable_thinking=false` the same block emits both tags, so the paired regex
would work; the defect only bites in the thinking-enabled configuration this run
used.

Why it is hard to see:

- the reward is still a number, still in a sane range, and still rises
  (`rollout/raw_reward` 0.324 → 0.617 over the frozen 717 steps);
- response length (~8k tokens) is expected under thinking mode, so it is not a
  symptom either;
- nothing in the logs records what the judge received — miles logs the reward,
  not the judge's request length;
- every other setting is ordinary: `kl-loss-coef 0.001`, `temperature=0.0`,
  `max_tokens=4096`, a reason-first judge template.

The single strongest piece of evidence against "the run is fine" is
reproducing the helper on a response of the shipped shape:

```
python3 -c 'import re; f=lambda t: re.sub(r"<think>[\s\S]*?</think>","",t).strip();
print(f("…reasoning…</think>final answer"))'
```

which returns the input unchanged. A runnable reproduction of the helper and the template
prefill is in `evaluations/issue/reference/evidence.md`.

## Reference solution

- **issue**: a passing diagnosis must name `extract_non_reasoning_content`, state
  that it relies on a paired tag, and connect that to the chat template
  prefilling the opening `<think>` into the prompt so the response carries only
  the closing tag. Blaming the judge model, response length, truncation, the KL
  coefficient, or concluding "training did not improve" must fail.

**Tier decision: issue-only.** No solution tier is shipped: the recorded
evidence comes from an internal image, and no rerun of either arm exists from
which a threshold could be calibrated.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle. The
  canonical judge requires verdict = 1.0 and both root-cause and evidence scores
  > 0.8; the reported score still uses the 0.2 / 0.6 / 0.2 weights.

Rollout dumps for the recorded job were deleted before packaging, so the
original sample responses — and with them the recorded judge-input lengths —
are not shipped; the mechanism is reproducible from the shipped code, and the
per-step trace under `workspace/outputs/` shows the run's ordinary-looking
curve.
