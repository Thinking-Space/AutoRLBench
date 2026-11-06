# Retrieval observations enter the loss and lose their passage bodies

Training Qwen2.5-3B with GRPO on Search-R1 nq_hotpotqa_train (NQ + HotpotQA merged, multi-turn retrieval-augmented). Retrieved `<information>` tokens are appended with `loss_mask=1` so the policy is trained on environment output, and `_passages2string` renders document titles only — the search channel is broken on both the loss side and the input side, and held-out NQ exact-match stalls at 16.7% against a 40.8% healthy protocol baseline.

| | |
| --- | --- |
| **Author** | Yinghao Chen |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | Reward |
| **Domain** | Search RL |
| **Tags** | `observation-loss-leak` `title-only-retrieval` `response-collapse` `silent-failure` |
| **Framework** | slime (GRPO), Megatron train + SGLang rollout |
| **Run time** | ~1.5 hours |
| **Resources** | 8 GPU |

## Difficulty

Two independent defects, both in `examples/search-r1/generate_with_search.py`. First, `generate` appends the retrieved `<information>` tokens with `loss_mask=1`, overriding the `trainable=False` that `append_response_tokens(..., trainable=False)` already encodes, and assigns the local list to `sample.loss_mask` — the policy gradient flows through environment output. Second, `_passages2string` renders each retrieved document as `Doc {idx+1}(Title: {title})` and never appends the passage body `text` it splits off the document `contents` two lines above, so every observation carries titles only.

Nothing fails loudly: training reward is present and the held-out score rises from 3.3% to 16.7% over 200 steps (21 eval points). The single strongest piece of evidence: train response length collapses from ~1038 tokens at step 0 to ~10 tokens by step 50 — trained-on observation tokens push the policy to stop searching and answer immediately — while truncation falls from ~38% to ~0%, so the held-out gap is not a truncation artifact. The healthy twin run of the same recipe (`tasks/slime_search_r1`) reaches 40.8% at the same horizon.

| step | 0 | 49 | 99 | 149 | 199 |
| --- | --- | --- | --- | --- | --- |
| `eval/nq_test` (defective) | 0.033 | 0.136 | 0.153 | 0.164 | 0.167 |

## Reference solution

- **issue**: a passing diagnosis returns an unhealthy verdict and names **both** defects with their causal roles — observation tokens trained on (`loss_mask=1` on `<information>` text) and title-only retrieval rendering. Naming only one of the two is partial credit; attributions that must fail: the reward design, truncation, and an absence of learning (reward and eval both rise).
- **solution**: set the observation tokens out of the loss (`loss_mask += [0] * len(obs_tokens_ids)`) and include the passage body in the rendered observation (`Doc {idx+1}(Title: {title}) {text}`). Both are required: measured partial-repair states at the graded horizon reach only 0.154–0.224, below the pass bar.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden `oracle.md` on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing is a conjunctive gate, not the weighted sum: `verdict_score == 1.0` and `root_cause_score > 0.8` and `evidence_score > 0.8`.
- **solution**: the agent repairs the workspace and reruns `examples/search-r1/run_qwen2.5_3B.sh` at the fixed `--num-rollout 100` / `--eval-interval 10` budget (eval points 0, 9, …, 99); `score_mode` is `final` — `eval/nq_test` at the run's final eval point (step 99) must land in **[0.28, 1.0]**, pass is score 1. The 0.28 bar is the same one the healthy twin `slime_search_r1` uses, so the threshold is not a tell for which side of the pair a task is on. Calibration runs at the graded horizon: repaired 0.3733 (healthy control) and 0.3570 (a second repaired run, best over steps 0–99); the frozen dual-defect evidence 0.1533; partial-repair states 0.18533 (loss mask fixed, title-only left) and 0.15433 / 0.2237 (loss-mask-only states); untrained step 0 spans 0.033–0.048. Thinnest margin is 0.056 on the failing side; the passing side's closest run clears by 0.077. The rerun budget is deliberately shorter than the 200-step frozen run: every failing state except the fully dual-defect run is censored past step 99, so no threshold at step 199 can be shown to reject them — do not extend the graded horizon without first running the partial-repair states out to 200 steps.
