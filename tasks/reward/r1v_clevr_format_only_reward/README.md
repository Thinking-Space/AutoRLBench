# Format-only reward on CLEVR counting

Training Qwen2-VL-2B-Instruct with GRPO on CLEVR-CoGenT-A counting questions. The launch script passes only the output-format reward, so the training reward saturates at 0.97 while held-out accuracy rises to 0.842 and falls back to 0.656, near its starting value of 0.602.

| | |
| --- | --- |
| **Author** | Yaolun Zhang |
| **Affiliation** | Oregon State University |
| **Type** | Reward |
| **Domain** | Visual counting RL |
| **Tags** | `format-only-reward` `missing-correctness-term` `reward-accuracy-divergence` |
| **Framework** | R1-V (GRPO), TRL 0.14.0, transformers 4.50.0.dev0, DeepSpeed 0.15.4 |
| **Run time** | ~2.1 hours (39 minutes training, the rest per-checkpoint evaluation) |
| **Resources** | 1 × H100 NVL (94 GB) |

## Difficulty

The defect is one token in `workspace/run_grpo.sh`: line 14 sets the reward
list to `sformat`, and line 29 passes it as `--reward_funcs`. `sformat` maps to
`strict_format_reward` in `workspace/open_r1/grpo.py`, which returns 1.0 when
the completion is exactly `<think>...</think><answer>integer</answer>` and
never reads the ground-truth count. The two registered answer rewards, `count`
and `accuracy`, are not used. The trainer, the data, the evaluation script and
the remaining hyperparameters (`lr=1e-6`, `beta=0.04`, 8 generations, 150
steps) are the same as in the comparison run below.

Nothing fails loudly. The run completes 150/150 steps, the logged reward climbs
from 0.715 (steps 1-15) to 0.972 (steps 16-150), KL stays small, and held-out
accuracy does improve for the first 75 steps, because a model that learns to
close its answer tag gets more answers parsed. An agent that reads only the
training log sees a healthy run; one that reads only the final accuracy sees
"slightly better than the base model". The strongest evidence is the
disagreement between the two: from step 75 to step 100 the training reward
stays at 0.97 while held-out accuracy drops from 0.842 to 0.584.

The loudest symptom points the wrong way. At step 100, 139 of 500 held-out
outputs run into the 256-token limit while listing objects in a loop, which
invites "raise the token limit" or "decoding problem". The same limit and
decoding give 6 such outputs at step 75.

| step | 0 | 25 | 50 | 75 | 100 | 125 | 150 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| held-out accuracy (500) | 0.602 | 0.744 | 0.828 | 0.842 | 0.584 | 0.700 | 0.656 |
| out-of-distribution accuracy (1000) | 0.103 | 0.243 | 0.199 | 0.221 | 0.192 | 0.198 | 0.187 |
| outputs without `</answer>` (held-out) | 42 | 31 | 1 | 6 | 139 | 60 | 76 |

Comparison run, not shipped: the same script with `REWARD_FUNCS=count,sformat`
and checkpoints every 50 steps, same seed, data, model and step budget.
Held-out accuracy is 0.982 / 0.974 / 0.946 at steps 50 / 100 / 150 and
out-of-distribution accuracy is 0.433 / 0.274 / 0.260. One run per side, so
these are margins, not variance: 29.0 points on held-out accuracy at step 150
and 7.3 points out of distribution.

## Reference solution

- **issue**: a passing diagnosis names the reward selection in
  `run_grpo.sh` (`--reward_funcs sformat`), states that
  `strict_format_reward` does not compare the answer with the ground truth, and
  connects this to the divergence between a saturated training reward and
  held-out accuracy that ends near its step-0 value. Describing a reward
  problem without saying that correctness is not rewarded earns partial
  root-cause credit (0.3) and cannot pass. Attributions that must fail:
  "training did not learn", learning rate, KL coefficient, group size, step
  budget, the 256-token limit, a bug inside either reward function, the
  evaluation script, distribution shift, and missing early stopping.

## Verification

- **issue**: an LLM judge compares `$AUTORLBENCH_RESULTS/agent_answer.md` with
  the hidden `oracle.md` on verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing is a conjunctive gate, not the weighted sum:
  `verdict_score == 1.0` and `root_cause_score > 0.8` and
  `evidence_score > 0.8`. Issue evaluation launches no training and needs no
  GPU.

Issue-only: the comparison run above is a single seed per side, and no
threshold has been calibrated on a harness rerun yet.

## Packaging notes

**Origin.** The failure was first observed in an autonomous research-agent run
on a CLEVR counting task, where the agent tried a format-only reward as one of
its experiments. For this task the experiment was re-run from the public R1-V
code in the shared image with the same reward, step budget, learning rate and KL coefficient; the shipped
logs and outputs are from that re-run, not from the agent's run.

**Assets.** Model: `Qwen/Qwen2-VL-2B-Instruct` @
`895c3a49bc3fa70a340399125c650a463535e71c` (Apache-2.0). Datasets:
`leonardPKU/clevr_cogen_a_train` @ `d5a8da66aa97c9c19ebe4075b7c7f777fef850a6`
and `RyanWW/Super-CLEVR` @ `2ecf1d00d1fa06b4a4ed9330269ac57d2f411090` (MIT).
`environment/build_data.py` derives `data/train` (68,000 rows), `data/val`
(2,000 rows) and `data/ood_val` (1,000 rows) from them. Code:
`StarsfieldAI/R1-V` @ `e35f97e566357631a2602accdd79610b752d8f0b` (Apache-2.0)
plus the edits in `environment/source_manifest.json`.

**Python layer.** The image has transformers 4.49.0 and no TRL, DeepSpeed,
qwen-vl-utils or math-verify. `prepare.sh` installs pinned versions into
`$AUTORLBENCH_PREPARED/pysite`; the transformers archive is fetched by URL and
checked against its SHA-256 and byte size.

**Image environment.** The image sets `MASTER_PORT=53394`, which
`run_grpo.sh` passes to `torchrun`. Two runs on one host collide on that port;
set `MASTER_PORT` per run when running side by side.

**Evaluation log.** The recorded job reached its wall-time limit while
evaluating checkpoint-125. A second invocation of `eval_checkpoints.py`
(`--steps 125,150`) finished the series, so `results/eval.log` has two headers
and `results/eval_metrics.jsonl` has the step-125 held-out line twice with
identical values.
