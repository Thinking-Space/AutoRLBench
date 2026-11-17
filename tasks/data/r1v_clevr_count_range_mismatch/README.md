# Single-question training set: the policy stops reading the question

Training Qwen2-VL-2B-Instruct with GRPO on CLEVR-CoGenT-A counting questions. Every training row asks the same question with answers 3-10, so the policy learns to return the total object count whatever is asked: out-of-distribution accuracy peaks at 0.675 at step 100 and falls to 0.251 at step 400 while in-distribution accuracy ends at 0.980.

| | |
| --- | --- |
| **Author** | Yaolun Zhang |
| **Affiliation** | Oregon State University |
| **Type** | Data |
| **Domain** | Visual counting RL |
| **Tags** | `single-question-template` `answer-range-mismatch` `question-ignored` `ood-decay` |
| **Framework** | R1-V (GRPO), TRL 0.14.0, transformers 4.50.0.dev0, DeepSpeed 0.15.4 |
| **Run time** | ~3.0 hours (83 minutes training, the rest per-checkpoint evaluation) |
| **Resources** | 1 × H100 NVL (94 GB) |

## Difficulty

The defect is in the data, not in any file an agent can diff.
`workspace/data/train` has 68,000 rows that all ask "How many items are there
in the image?", with answers 3 to 10 in equal shares. `workspace/data/ood_val`
asks 995 distinct Super-CLEVR questions over 1,000 rows, and 864 of its answers
are 0, 1 or 2. `workspace/data/val`, the in-distribution held-out set, has the
training set's single question and answer range, so it cannot see the problem.
Code, rewards (`count` + `sformat`), and hyperparameters (`lr=2e-6`,
`beta=0.01`, 8 generations, 400 steps) are the trainer's ordinary recipe.

Nothing fails loudly. The run completes 400/400 steps, the exact-match training
reward stays above 0.93 after step 50, and in-distribution accuracy ends at
0.980. The out-of-distribution curve first improves sixfold, which reads as
success, and then decays, which reads as ordinary over-training and invites
"lower the learning rate", "raise the KL coefficient" or "stop at step 100".
Those change the speed, not the direction. The evidence that pins the cause is
in the per-item outputs: the number of out-of-distribution predictions in the
training answer range 3-10 grows from 225 at step 100 to 626 at step 400,
against 136 gold answers in that range, and late outputs list every object in
the scene and return the total without addressing the question.

| step | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| out-of-distribution accuracy (1000) | 0.103 | 0.619 | 0.675 | 0.528 | 0.346 | 0.208 | 0.239 | 0.243 | 0.251 |
| in-distribution accuracy (500) | 0.602 | 0.964 | 0.966 | 0.930 | 0.912 | 0.972 | 0.978 | 0.978 | 0.980 |
| OOD predictions in 3-10 | 350 | 206 | 225 | 343 | 496 | 612 | 575 | 612 | 626 |
| OOD accuracy, gold 0-2 (864) | 0.094 | 0.654 | 0.734 | 0.569 | 0.365 | 0.219 | 0.256 | 0.260 | 0.272 |

A second run, not shipped, used half the learning rate and four times the KL
coefficient (`lr=1e-6`, `beta=0.04`, 150 steps, same seed and data). It shows
the same movement: out-of-distribution accuracy 0.433 / 0.274 / 0.260 at steps
50 / 100 / 150, with predictions of 3 or more growing from 395 to 583 of 1,000.
One run per setting.

## Reference solution

- **issue**: a passing diagnosis names both parts: (1) the training set asks a
  single question with answers 3-10, while the out-of-distribution held-out
  set asks varied questions whose answers are mostly 0-2; and (2) the chain to
  the failure: with one question the prompt text carries no reward
  information, the exact-match reward only reinforces "output the total object
  count", and the policy applies that to every question, so out-of-distribution
  predictions move into the 3-10 range and accuracy falls from 0.675 to 0.251.
  Stating only "distribution shift" or only "over-training" earns partial
  root-cause credit (0.3) and cannot pass. Attributions that must fail: a
  defect in either reward function, learning rate or KL coefficient as the
  root cause, missing early stopping as the root cause, "training did not
  learn", the evaluation script or token limit, and trainer or prompt-template
  bugs.

## Verification

- **issue**: an LLM judge compares `$AUTORLBENCH_RESULTS/agent_answer.md` with
  the hidden `oracle.md` on verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing is a conjunctive gate, not the weighted sum:
  `verdict_score == 1.0` and `root_cause_score > 0.8` and
  `evidence_score > 0.8`. Issue evaluation launches no training and needs no
  GPU.

Issue-only: there is no repaired run. A repair changes the training data, and
the instruction fixes the dataset, so a rerun threshold would not grade it.

## Packaging notes

**Origin.** The pattern was first observed in autonomous research-agent runs
on a CLEVR counting task, where out-of-distribution accuracy peaked early and
decayed in several independent attempts. For this task the experiment was
re-run from the public R1-V code in the shared image; the shipped logs and
outputs are from that re-run.

**Assets.** Model: `Qwen/Qwen2-VL-2B-Instruct` @
`895c3a49bc3fa70a340399125c650a463535e71c` (Apache-2.0). Datasets:
`leonardPKU/clevr_cogen_a_train` @ `d5a8da66aa97c9c19ebe4075b7c7f777fef850a6`
and `RyanWW/Super-CLEVR` @ `2ecf1d00d1fa06b4a4ed9330269ac57d2f411090` (MIT).
`environment/build_data.py` derives `data/train` (68,000 rows), `data/val`
(2,000 rows) and `data/ood_val` (1,000 rows) from them; the evaluation uses
the first 500 rows of `val` and all of `ood_val`. Code: `StarsfieldAI/R1-V` @
`e35f97e566357631a2602accdd79610b752d8f0b` (Apache-2.0) plus the edits in
`environment/source_manifest.json`.

**Python layer.** The image has transformers 4.49.0 and no TRL, DeepSpeed,
qwen-vl-utils or math-verify. `prepare.sh` installs pinned versions into
`$AUTORLBENCH_PREPARED/pysite`; the transformers archive is fetched by URL and
checked against its SHA-256 and byte size.

**Image environment.** The image sets `MASTER_PORT=53394`, which
`run_grpo.sh` passes to `torchrun`. Two runs on one host collide on that port;
set `MASTER_PORT` per run when running side by side.

**Sibling task.** `reward/r1v_clevr_format_only_reward` uses the same code,
model and data with a different launch configuration and a different defect.
The two runs share no artifacts.
