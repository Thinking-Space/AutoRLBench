# Recorded comparison

Four 60-step runs of `run_tool_gsm_hard.sh`. All four use the same image, model, data split,
budget, reward, tool and launch script.

- **Defective arm:** the shipped workspace.
- **Control:** the same workspace with `fix.diff` applied. That removes the call to
  `shorten_numeric_literals` in `verl/experimental/agent_loop/tool_agent_loop.py` and the helper in
  `verl/experimental/agent_loop/utils.py`.
- **Seeds:** one run per arm per seed (42, 43).
- **Shipped evidence:** the seed-42 defective run (`workspace/results/tool_gsm_hard_seed42`).
- **Files next to this one:** the `metrics.jsonl` and `val_generations/` of the other three runs.

**Held-out accuracy** on the 253 held-out GSM-Hard problems, greedy decoding. This is the reward's
`acc`, which `val-core/gsm_hard/acc/mean@1` logs.

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | mean 20-60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 (shipped) | 0.3281 | 0.4348 | 0.5099 | 0.5692 | 0.5771 | 0.5810 | 0.5968 | 0.5668 |
| defective, seed 43 | 0.3281 | 0.4111 | 0.4743 | 0.5020 | 0.5375 | 0.5336 | 0.5652 | 0.5225 |
| control, seed 42 | 0.4585 | 0.5336 | 0.6561 | 0.6482 | 0.6561 | 0.6957 | 0.6561 | 0.6625 |
| control, seed 43 | 0.4466 | 0.5020 | 0.6364 | 0.6680 | 0.6364 | 0.6403 | 0.6917 | 0.6545 |

- **Separation from step 20 on, all four runs:** lowest control evaluation 0.6364, highest
  defective evaluation 0.5968 (separated: yes).
- **Seed-paired gaps:** 5.9 and 12.6 points at step 60, and 9.6 and 13.2 points on the steps 20-60
  mean.
- **Granularity:** one held-out question is 0.4 points.
- **Label noise:** GSM-Hard labels are noisy. Some golds are negative or fractional counts, because
  the dataset swaps in large numbers without re-checking the story. Both arms train and evaluate on
  the same labels.

**Held-out tool-call rate**: the share of responses containing `<tool_call>`.

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 | 0.466 | 0.067 | 0.036 | 0.024 | 0.016 | 0.008 | 0.008 |
| defective, seed 43 | 0.451 | 0.103 | 0.051 | 0.028 | 0.028 | 0.012 | 0.016 |
| control, seed 42 | 0.458 | 0.510 | 0.751 | 0.767 | 0.798 | 0.925 | 0.889 |
| control, seed 43 | 0.431 | 0.510 | 0.810 | 0.854 | 0.862 | 0.779 | 0.941 |

**Training metrics** (seed 42 / seed 43):

| metric | defective | control |
| --- | --- | --- |
| `critic/score/mean` step 1 → 60 | 0.363 → 0.631 / 0.398 → 0.564 | 0.479 → 0.654 / 0.490 → 0.646 |
| `num_turns/mean` step 1 → 10 → 60 | 3.17 → 2.31 → 2.01 / 3.13 → 2.30 → 2.04 | 3.14 → 2.98 → 3.84 / 3.00 → 3.04 → 3.94 |
| `actor/kl_loss` at step 60 | 0.083 / 0.073 | 0.082 / 0.080 |
| `response_length/mean` at step 60 | 264 / 295 | 369 / 329 |

**Why issue-only.** The closest seed-paired step-60 gap is 5.9 points on 253 problems.
That is too thin for a rerun threshold.

**Runtime.** Apptainer 1.4.0 (`apptainer exec --nv`) on a SIF built from the declared digest
`verlai/verl@sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6`, on
1 × A100-SXM4-80GB, with the workspace mounted at `/workspace`. Each pair of runs shared the node.

**Provenance.**
- The same rewriting function was first recorded as a helper inside `tools/python_tool.py`, two
  seeds. The defective runs averaged 0.5692 and 0.5534 over steps 20-60, and tool use fell to 0.00.
- To keep the tool file honest, the task ships the agent-loop placement with its own two runs.
  Those runs are the defective rows above.
- The control runs are unchanged. Two earlier seed-43 attempts crashed during an unrelated node
  overload; they were discarded, not spliced.
