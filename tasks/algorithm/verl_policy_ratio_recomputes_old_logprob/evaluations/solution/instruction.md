# Repair the training experiment

Investigate the unhealthy verl experiment in `/workspace`, then repair the source
or configuration. There is no source-file whitelist. Preserve unrelated behavior.
Explain the root cause, evidence, and repair in `/workspace/report.md`.

Use the fixed official GSM8K train/test data, model, seeds, batch sizes, rollout
count, lengths, optimizer passes, learning rate, 30-step budget and evaluation
schedule. This recipe uses 16 prompts per step,
8 generated responses per prompt and 4 actor passes per rollout.
Do not change checkpoint/resume steps or disable configured losses.
Keep `observed_run/` unchanged. Do not modify evaluation, oracle or verifier files.
Do not copy, edit, fabricate or replay runtime evidence. Do not access the network.

Do not launch training or produce runtime evidence yourself. After you submit the
repair and `report.md`, the evaluation harness runs the following command from
`/workspace` with a 7200-second training budget:

```bash
bash scripts/train.sh --output "$AUTORLBENCH_RESULTS/experiment_logs"
```

The harness clears the final output directory before training. The verifier
checks the resulting `completed.json`, JSONL evidence, resolved configuration,
and training artifacts. Agent-generated logs are not accepted as a training run.
Final accuracy at step 30 must be strictly greater than 45%: at least 46 correct
answers out of the fixed 100 questions.
Acceptance requires the exact final-step held-out evaluation and the task's
runtime mechanism invariants; step-0, earlier-best or partial logs do not count.
