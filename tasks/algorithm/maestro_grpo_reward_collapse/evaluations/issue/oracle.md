# Canonical issue oracle

Reconciled from the frozen evidence and the diff between `0207b9a9f95eee575eaf829ca95ce20d39e4672b` and `797c70b77930d4f975b004a096745a77b0a5438a`.

Expected verdict: **unhealthy**.

## Root cause

- `verl_tool/agent_loop/verltool_agent_loop.py:VerlToolAgentLoop.run`, first-turn search reconstruction
  - kind: `symbol`
  - observed: `gen_text = prefix_text + search_content`; the reconstructed sequence has no terminal EOS, but the next block uses `new_gen_tokens[:-1] + gen_ids[-1:]` as though it did. With this tokenizer `</search>` ends in a separate `>` token, so the closing tag becomes `</search<|im_end|>`.
  - expected: append `"<|im_end|>"` when constructing `gen_text`, so the replacement removes only the synthetic EOS and preserves the full closing search tag.

This is one root cause spanning the reconstruction and its downstream slice, not two independent defects. A generic statement that EOS is missing, without connecting it to deletion of the closing delimiter, is incomplete attribution.

## Observable impact

Every item in this section is derivable from the shipped workspace
(`results/recorded_run/`: `metrics/*.json`, the three-row-per-step sampled
`rollouts/*.jsonl` described by `rollout_sampling.json`, and
`logs/train_4_gpus.log.gz`). The historical full-batch tag counts and the
counterfactual control-run figures are author-side reference and live in
Documentation reconciliation below. Each bullet is one item.

- Malformed search tags in the sampled step-1 rollouts (`rollouts/1.jsonl`):
  the generated outputs have unequal opening/closing search-tag counts, and
  the closing tag appears as `</search<|im_end|>` — the `<|im_end|>` from the
  first-turn reconstruction eats the closing `>` (degenerating further to
  forms like `</search<<|im_start|` by step 200 in `rollouts/200.jsonl`).
- Step-200 collapse (`metrics/0200.json`): `critic/score/mean=-1.0`,
  `verl_tool/is_traj_finished/mean=0.0`, and responses clipping at the
  4096-token cap (`response_length/clip_ratio=0.5625`,
  `response_length/max=4096`) with no completed final-answer trajectory among
  the sampled rows.
- Training reward does improve temporarily (step 120 mean 0.7109375,
  `metrics/0120.json`). Therefore "training never learned" is a wrong
  attribution.

## Trigger conditions

- Multimodal asynchronous GRPO rollouts enter the step-0 branch with a complete search block.
- Re-encoding is accepted by the `len(new_gen_tokens) <= len(gen_ids)` branch.
- The code substitutes the final token before appending the action to the trajectory and decoding it for tool interaction/reward computation.

## Documentation reconciliation

The one-line Git diff establishes the root cause; reward anomalies, malformed rollout tags and truncation establish the observed impact. GPU initialization workarounds, service ports, run names and resume-mode labels are not the defect. The control configuration uses `resume_mode=auto` but its log explicitly says `Training from scratch`; the defective run disables resume. The same data seed and effective training settings were used. Neither a new reward design nor changes to the data are required to repair this delimiter bug.

Author-side reference, not agent-derivable and not part of the evidence
denominator: in the original full step-1 batch 124/128 outputs have unequal
opening/closing search-tag counts against the hidden control's 7/128
(preserved in the solution oracle's `rollout_full_batch_summary.json`; the
shipped `rollouts/1.jsonl` carries only three sampled rows), all 128
defective step-200 trajectories stop at the response-length limit, and the
control run has mean training score 0.6796875 with no clipped responses,
completion mean 1.0, and 128/128 final-answer trajectories.
