# Recorded comparison

Four 60-step runs of `run_dapo_gsm8k.sh`, all with the same image, data, model, budget and launch
script. The defective arm is the shipped workspace. The control is the same workspace with
`fix.diff` applied; only `verl/trainer/ppo/ray_trainer.py` differs, by one line. Each arm ran once per
seed (42, 43). The shipped evidence is the seed-42 defective run
(`workspace/results/dapo_gsm8k_seed42`). The `metrics.jsonl` files of the other three runs sit next to
this file.

Held-out accuracy is measured on the 1,319 GSM8K test questions with greedy decoding, from
`val_generations/<step>.jsonl`. Each cell gives two numbers:
1. the verl reward's `acc`, which is what `val-core/openai/gsm8k/acc/mean@1` logs (flexible
   extraction, string match against the gold);
2. a numeric re-score: the last number in the response must equal the gold numerically, so "18.00"
   counts for "18".

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| defective, seed 42 (shipped) | 0.4382 / 0.4701 | 0.4466 / 0.4731 | 0.4640 / 0.4860 | 0.4435 / 0.4708 | 0.4564 / 0.4761 | 0.4602 / 0.4814 | 0.4647 / 0.4845 |
| defective, seed 43 | 0.4284 / 0.4579 | 0.4450 / 0.4738 | 0.4610 / 0.4807 | 0.4792 / 0.5027 | 0.4534 / 0.4723 | 0.4572 / 0.4845 | 0.4556 / 0.4792 |
| control, seed 42 | 0.4329 / 0.4632 | 0.4587 / 0.4701 | 0.4822 / 0.4852 | 0.5201 / 0.5201 | 0.5254 / 0.5254 | 0.5262 / 0.5262 | 0.5451 / 0.5459 |
| control, seed 43 | 0.4261 / 0.4534 | 0.4670 / 0.4746 | 0.4761 / 0.4776 | 0.5216 / 0.5216 | 0.5178 / 0.5178 | 0.5284 / 0.5292 | 0.5171 / 0.5171 |

Step-0 values differ by up to 0.012 between identical starts. vLLM's batched greedy decoding is not
deterministic.

Training metrics over steps 1-60, given as seed 42 / seed 43:

| metric | defective | control |
| --- | --- | --- |
| `critic/score/mean` max | 0.465 / 0.455 | 0.758 / 0.778 |
| `actor/entropy` at step 60 | 0.50 / 0.55 (reference model's entropy, see oracle) | 0.10 / 0.11 |
| `actor/kl_loss` max | 0.0023 / 0.0025 | 0.085 / 0.070 |
| `actor/ppo_kl` range | −0.0019 to +0.0000003 / −0.0023 to −0.00002 | −0.00012 to +0.00024 / −0.00015 to +0.00019 |
| `actor/pg_clipfrac`, steps 10-60 | 0.0065-0.0118 / 0.0070-0.0128 | 0.0005-0.0016 / 0.0005-0.0013 |
| `actor/grad_norm` range | 0.74-1.05 / 0.79-0.98 | 0.68-1.10 / 0.63-0.95 |

**Why issue-only.** At step 60 the numeric re-score puts the defective runs at 0.4792-0.4845 and
the controls at 0.5171-0.5459. That is a gap of 3.3-6.7 points from single runs. Within the
same runs the arms come closer: a defective run reached 0.5027 at step 30, and control seed 43
dropped from 0.5292 to 0.5171 between steps 50 and 60. Any threshold would have only a few points
of margin on each side, which is too thin to grade a rerun reliably.

**Runtime.** Apptainer 1.4.0 (`apptainer exec --nv`) on a SIF built from the declared digest
`verlai/verl@sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6`, on
1 × A100-SXM4-80GB, with the workspace mounted at `/workspace`. The shipped run took about
26 minutes (02:39-03:05 by its log timestamps), while a second run shared the node's CPUs.

**Provenance.**
- Before these runs, the same stall was recorded with the substitution placed in
  `verl/workers/utils/losses.py::ppo_loss` (ratio taken against `ref_log_prob`): two seeds, ending
  at 0.4541 and 0.4526.
- The task ships the trainer-level placement and its own runs instead. Those runs are the ones in
  the tables above.
