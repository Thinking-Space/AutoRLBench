# MoE auxiliary gradients miss token normalization

Training Qwen/Qwen1.5-MoE-A2.7B-Chat with GRPO on GSM8K. The per-token loss
callback omits the token count that MCore needs to normalize the native MoE
auxiliary losses, so the z-loss gradients are never divided back.

| | |
| --- | --- |
| **Author** | AutoRLBench |
| **Domain** | infrastructure |
| **Tags** | `verl` `grpo` `moe` `gradient-normalization` |
| **Model** | Qwen/Qwen1.5-MoE-A2.7B-Chat |
| **Dataset** | openai/gsm8k main; 7473 train rows, 64 held-out validation rows from the official 1319-row test split |
| **Framework** | verl, Megatron engine, `scripts/train.sh` |
| **Environment** | shared `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image · `prepare.sh` stages the patched MCore router into `$AUTORLBENCH_PREPARED` · 2 GPUs with at least 56 GiB free per card · ~600 GiB host RAM (optimizer/master states are fully CPU-offloaded) |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective + 1 repaired run, 150 steps each |

## Difficulty

`MegatronEngineWithLMHead` in
`verl/workers/engine/megatron/transformer_impl.py` returns a two-item
`(loss, metrics)` callback while per-token loss, CP2 and a nonzero
`moe_z_loss_coeff` are enabled, so MCore receives no token count for final
normalization. The router z-loss gradients (pre-scaled by `num_tokens`) are
never divided back: `actor/grad_norm` is ~223 at step 1 against ~3.3 for the
repaired callback. The common router pre-scaling backport and padding-mask
compatibility fix are already present, which rules out the two nearby
explanations. The two supplied workspaces have identical training
configurations, launch scripts and environments; their only training-code
difference is the callback repair.

| Held-out accuracy (64 questions) | Defective | Repaired |
| --- | ---: | ---: |
| Step 100 | 0% | 45.3125% |
| Step 150 | 0% | 54.6875% |

Defective W&B history ends at step 148, but console history and final
validation reach 150; repaired W&B reaches 150. The histories remain
separate, no missing records are synthesized, and one run per arm supports
the observed gap without establishing cross-seed variance.

## Reference solution

- **issue**: identify the missing native `(loss sum, token count, metrics)`
  contract and explain how it skips final normalization of z-loss gradients.
- **solution**: return real routed-token counts, including prompt tokens, with
  consistent DP/CP policy scaling, then complete a fresh 150-step run. The
  reference patch, repaired source and calibration curves are hidden in the
  solution oracle.

## Verification

- **issue**: an LLM judge scores verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing requires `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`; the weighted total alone is not sufficient.
- **solution**: the agent repairs the workspace and submits `report.md` without
  training; a missing report scores 0. The harness then reruns
  `scripts/train.sh` per `run_spec.json` and requires
  `val-core/openai/gsm8k/acc/mean@1` at step 150 ≥ 0.4 (cap 1.0). Only the
  step-150 evaluation of a fresh run counts: the grader reads only
  `experiment_logs`, rejects step-0-only or earlier-best logs, and extending
  the run or raising eval frequency cannot lift an unrepaired score. The
  grader's `required_log_patterns` additionally anchor the trigger
  configuration — per-token loss enabled, CP2, EP2 and `moe_z_loss_coeff`
  0.001 must each appear in the fresh run's text logs — and all accuracy
  values must stay in [0, 1], so a repair that avoids the defective code path
  cannot pass. The threshold sits 0.4 above the defective run and 0.146875
  below the repaired one; the lowest observed repaired point clears it by
  0.053125. It is calibrated on the recorded run per arm and corroborated by
  a same-configuration rerun on the shared image (2026-11-03): defective 0/0
  in both runs, repaired 0.453125/0.546875 recorded and 0.515625/0.546875
  rerun. Neither run records step-0 validation, and the two runs are not
  deterministic replays, so treat scores near 0.4 as untested territory.

## Environment notes

The task runs on the shared `verl` image
(`docker/images/verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12.Dockerfile`,
the same pinned digest the former `verl-moe-token` image was built on; no
custom image is built or published). The MCore router prescaling backport
(`environment/mcore5047-minimal.patch`, shared by both arms and not the defect
under diagnosis) is applied by `environment/prepare.sh`: the `megatron`
package is copied out of the image into `$AUTORLBENCH_PREPARED/mcore`, patched
there, and takes effect via the `PYTHONPATH` line prepare writes to
`prepared/env`. GPU visibility comes from the task metadata (`gpus = 2`):
the docker backend passes `--gpus`, Harbor gets the synthesized compose
overlay. Weights, datasets and checkpoints are omitted. The supplied material
lacks W&B binaries and rollout sample dumps. Framework/new glue use Apache-2.0,
GSM8K uses MIT, and the model has its own license in `environment/model-LICENSE`.
