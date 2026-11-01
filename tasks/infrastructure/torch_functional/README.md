# verl_implementation diagnosis: torch_functional

## Case overview

- Domain: infrastructure
- Evaluation key: `verl_implementation/torch_functional`
- Case family: `verl_implementation`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-1.5B-Instruct`
- Training framework: `verl`
- Primary source evaluation metric: `val-core/taco/acc/mean@1`
- Environment: `verl` · 8 GPUs

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. Metric names preserve the source experiment;
tasks using the same held-out protocol may intentionally share a name.

## Canonical failure reason

`prepare_model_inputs` builds the log-prob recompute labels un-rolled —
`input_ids_rmpad_rolled = input_ids_rmpad` — instead of the causal roll
`input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)`.
These un-rolled labels are passed both to `logprobs_from_logits` and to the
fused-kernel `shift_labels`. `logprobs_from_logits` does a plain
log-softmax+gather (no internal shift) and the model's logits are standard
(`logits[i]` predicts `token[i+1]`), so without the roll every recomputed
response log-prob is misaligned by one position. This is the live
FSDP/NO_PADDING log-prob path (`old_log_prob`, `ref_log_prob`, actor-update
`log_prob`).

> Re-targeting note (2026-10-05): two earlier framings were inert on this
> engine and have been reverted. (1) The original injection in
> `verl/utils/torch_functional.py::log_probs_from_logits_response` is dead code
> in verl 0.9.0.dev (zero call sites). (2) A second re-target to
> `verl/workers/utils/padding.py::no_padding_2_padding`'s left-shift slice also
> could not break/recover training, because that slice is only the *consumer*
> of a rolled array — the real causal shift lives in the engine's
> `torch.roll`. A 4-way check on the devspace (Qwen2.5-1.5B) confirmed that only
> `roll + no_padding_2_padding[-1]-slice` reproduces textbook log-probs
> (mean |Δ| = 0.0000); dropping the roll yields |Δ| ≈ 10 nats regardless of the
> padding slice. The defect now lives in the live `prepare_model_inputs` roll;
> `torch_functional.py` and `padding.py` are restored to their healthy upstream
> form. A full 400-step run with the roll restored trained healthily
> (`rollout_corr/kl` ≈ 0, val peak 0.125); dropping the roll collapses val and
> reward to 0.

## Workspace and hidden evaluations


| Path                    | Visibility and purpose                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `workspace/`            | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/`    | Neutral diagnosis instruction, issue oracle, and task-local issue verifier.                |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier.           |


The agent cannot read oracle or verifier paths. `issue` grading compares
`$AUTORLBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reads only the new experiment logs.

## Source and preserved context

The following source and historical packaging fields were moved from
`metadata.yaml` so the machine metadata remains aligned with the core
schema. They are preserved here without changing their meaning.

```yaml
bench_category: verl_implementation
source_task: data_call/tasks/reward_design/task_function_call
evidence_kind: injected_implementation_defect
packaging_tier: B0
packaging_ready: true
b1_scoring_ready: true
target_files: verl/workers/engine/fsdp/transformer_impl.py
validation_stack: TACO_EASY
already_at_baseline: false
metric_status: log_reconstructed
verify:
  num_gpus: 8
  timeout_minutes: 660
  max_train_steps: 400
  eval_metric_key: val-core/taco/acc/mean@1
```

The package was copied from `task_function_call`. `taco.py` was restored from
`lyh` branch `tasks/verl_coding/tasks_coding_pass_01`. The injected defect is
the dropped causal roll in
`verl/workers/engine/fsdp/transformer_impl.py::prepare_model_inputs`
(the earlier `log_probs_from_logits_response` and `no_padding_2_padding`
framings were inert on this engine and have been reverted — see the
re-targeting note above). `workspace/logs` carries a genuine 400-step collapse
run of this defect (`qwen2_5_1.5b_instruct_taco_fsdp_20260910_040027.log` plus
the matching wandb offline datastore `run-xr5gf82b.wandb`): it runs the live
FSDP/NO_PADDING path (`strategy=fsdp`, `use_remove_padding=True`, `_v1=False`)
with the roll dropped, so `rollout_corr/kl` holds ≈ 20–22 throughout,
`val-core/taco/acc/mean@1` falls 0.071 → 0.0 by step 150, `critic/rewards/mean`
collapses to 0, and `response_length/mean` balloons from ~168 to ~2400. This is
the diagnostic evidence exposed to the agent, not a healthy baseline.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `verl/workers/engine/fsdp/transformer_impl.py::prepare_model_inputs`

Required fix targets:

- `verl/workers/engine/fsdp/transformer_impl.py::prepare_model_inputs`: `restore`;
  target state/value:
  `input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)`.

Documentation reconciliation:

- `verl/workers/utils/padding.py::no_padding_2_padding` keeps its healthy
  `values[seq_offset - resp_len - 1 : seq_offset - 1]` left-shift slice — it is
  the consumer of the rolled array, not the defect.
- `taco.py` matches the healthy `tasks_coding_pass_01` scorer and is not causal.
- Parquet, launch budget, and the 1.5B checkpoint stay fixed.

## Evaluation design

`issue` grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and `root_cause_score` and `evidence_score` each > 0.8; the weighted total alone is not a pass condition.
The essential root cause is the dropped causal roll in
`prepare_model_inputs` (`input_ids_rmpad_rolled = input_ids_rmpad` instead of
`torch.roll(input_ids_rmpad, shifts=-1, dims=1)`).

`solution` does not grade a structured diagnosis. The agent repairs the staged workspace and submits `report.md` without
launching training. The harness runs the designated launch script for 400 steps
and produces fresh experiment logs. Scoring is weighted:

- validation: a finite `val-core/taco/acc/mean@1` peak of at least `0.10`
  scores `1`, otherwise `0`
- training reward: the mean `critic/rewards/mean` over training steps must be
  strictly above `0.10` to score `1`, otherwise `0`
- final `score = 0.8 * validation + 0.2 * training reward`

Pass requires both components (`score == 1.0`). The framework still records
the weighted float when only one component passes.

Thresholds are calibrated to the fixed-run reference on this engine
(devspace, 8×H100, 2026-10-05): with the roll restored, `val-core/taco/acc/mean@1`
rises from a step-0 baseline of `0.054` to a peak of `0.125` (step 150, final
`0.109`) and `critic/rewards/mean` averages ≈ `0.15`–`0.20`, while
`rollout_corr/kl` stays ≈ 0. With the roll dropped, validation and reward both
collapse to `0`. The `0.10 / 0.10` bars sit comfortably between the two regimes.

## External dataset asset

- Repository: `LIF1014/ptdbench-verl-implementation-torch-functional-dataset`
- Publication status: `published`
- Recorded revision: `83da7ccf4a74543f5aa3e46191c4916d7029564b`
- Dataset license: `Apache-2.0`
- Provenance: Same TACO EASY snapshot as `task_function_call` (8368 train / 184 test).


| Runtime path                            | Size              | SHA-256                                                            |
| --------------------------------------- | ----------------- | ------------------------------------------------------------------ |
| `workspace/taco_verl/test.parquet`      | 61,082,676 bytes  | `e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa` |
| `workspace/taco_verl/train.parquet`     | 606,016,648 bytes | `7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706` |


`environment/prepare.sh` downloads the dataset archive and verifies the two
extracted parquet files against the sizes and SHA-256 values above before
placing them in the workspace. The current whole-repository fetch does not
pass a revision; the extracted-file checks enforce the recorded dataset bytes.

## External model asset

- Repository: `Qwen/Qwen2.5-1.5B-Instruct` (official)
- Recorded revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Byte-identical, on every allowed file, to the former task snapshot
  `LIF1014/ptdbench-verl-implementation-torch-functional@36a4c22`; switched
  because that repo name is visible to the agent through the shared download cache.
- Model: `Qwen2.5-1.5B-Instruct` (`model.safetensors` SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`)

The current prepare script fetches the model without enforcing this revision
or checksum. Reproducible publication requires adding those checks; the
recorded identity above alone does not pin a fresh download.

## Environment requirement

- Base image: `verlai/verl:vllm017.latest`
- Use the Docker agent backend after external asset publication.
- The local backend does not build the task image or hydrate external assets.
- The designated experiment uses 8 GPUs, a 400-step budget, and a large shared-memory allocation.
- Launch script: `examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh`

The grader requires the final validation point and every training-reward step
from 1 through the declared horizon; partial or over-budget logs score 0.
This completion gate does not change the existing metric thresholds.
