# Slime Search-R1 GRPO run assessment: slime_search_r1_nomask

## Case overview

- Evaluation key: `slime_search_r1_nomask`
- Case family: `slime_implementation`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-3B`
- Dataset: `Search-R1 nq_hotpotqa_train` (NQ + HotpotQA merged)
- Training framework: `slime` (Megatron train + SGLang rollout)
- Recipe: `examples/search-r1/` — multi-turn retrieval-augmented GRPO
- Primary source evaluation metric: `eval/nq_test`
- Environment: `slime-v0.3.2` · 8 GPUs

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. Metric names preserve the source experiment;
tasks using the same held-out protocol may intentionally share a name.

## Canonical failure reason

Observation tokens are trained on (`loss_mask=1` on retrieved
`<information>` text); frozen `eval/nq_test` reaches only 16.5% at the
200-step horizon versus a healthy protocol baseline of 40.8%.

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local issue verifier. |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier. |

The agent cannot read oracle or verifier paths. `issue` grading compares
`$AUTORLBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reads only the new experiment logs.

## Source and preserved context

```yaml
evidence_kind: historical_run
packaging_ready: true
healthy_control: tasks/slime_search_r1
verify:
  num_gpus: 8
  max_train_steps: 200
  eval_interval: 10
  last_eval_step: 199
  eval_metric_key: eval/nq_test
historical_run:
  metric_status: wandb_extracted
  wandb_run_id: wi4fh8cz
  wandb_group: search-r1_qwen2.5-3B
  eval_first: 0.048
  eval_last: 0.16466666666666666
  eval_max: 0.16566666666666666
  n_eval_points: 21
  attempt: 1
healthy_control_at_horizon:
  wandb_eval_step: 199
  eval/nq_test: 0.4083333333333333
# The repair-and-rerun budget is shorter than the frozen run -- see
# "Evaluation design" for why grading stops at step 99.
solution_rerun:
  max_train_steps: 100
  last_eval_step: 99
  historical_at_step_99: 0.15433333333333332
  healthy_control_at_step_99: 0.37333333333333335
```

The workspace preserves the launch script, the custom multi-turn search
rollout with the injected observation-mask defect, the exact-match reward,
a compact frozen metrics package under `workspace/outputs/`, and a slimmed
offline W&B run (console `output.log` through train step 100, config,
summary; no `.wandb` binary history).

## Original failure evidence

- twenty-one reconstructed `eval/nq_test` points through `eval/step` 199 (the
  last eval in `--eval-interval 10` before the 200-step horizon) rise from
  about 4.8% to 16.5% (`eval_last` 0.1647; peak 16.6% at step 189)
- a healthy Qwen2.5-3B Search-R1 GRPO run (`tasks/slime_search_r1`) reaches
  about **40.8%** at the same horizon, and about 37.3% by step 99
- train response length collapses from about 1751 tokens at step 0 to about
  9 tokens by step 100: trained-on observation tokens push the policy to stop
  searching and answer immediately
- training `rollout/raw_reward` does not establish held-out health
- truncation falls from ~38% at step 0 to ~0% by step 100, so the held-out
  gap is not a truncation artifact
- `examples/search-r1/generate_with_search.py` appends retrieved
  `<information>` tokens with `loss_mask=1` while `append_response_tokens`
  still marks them `trainable=False`; the local list is assigned to
  `sample.loss_mask`

The preserved source metric is `eval/nq_test`. The frozen workspace remains
the evidence package for diagnosis. Solution grading requires the agent to
repair the workspace and submit `report.md`; the harness reruns training and
the verifier reads only the new
logs under `$AUTORLBENCH_RESULTS/experiment_logs`.

## Healthy protocol baseline

A healthy run of Qwen2.5-3B on Search-R1 nq_hotpotqa_train under the same
slime/GRPO recipe, with observation tokens masked out of the loss, reaches
about **40.8% at 200 steps** (about 37.3% by step 99). This case's frozen
`eval_last` 16.5% is well below that protocol baseline.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `observation_loss_mask` —
  `examples/search-r1/generate_with_search.py::generate`

Required fix targets:

- `observation_loss_mask`: `set` observation tokens out of the loss
  (`loss_mask += [0] * len(obs_tokens_ids)`).

Documentation reconciliation:

- Frozen W&B reconstruction uses `eval/step` 199 as the 200-step horizon
  because slime logs `eval/step = rollout_id` and evals at 0, 9, …, 199.
- Naming the observation loss mask with the correct causal role is a
  complete attribution.

## Evaluation design

`issue` grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and `root_cause_score` and `evidence_score` each > 0.8; the weighted total alone is not a pass condition.

`solution` does not grade a structured diagnosis. The agent must edit the
workspace and submit `report.md` without training. The harness runs
`examples/search-r1/run_qwen2.5_3B.sh` at the fixed
`--num-rollout 100` / `--eval-interval 10` budget, giving eval points 0, 9, …,
99. `score_mode` is `best`; the highest finite `eval/nq_test` over the run must
land in `[0.28, 1.0]`. Pass is score `1`; otherwise `0`. `0.28` is the same bar
the healthy twin `slime_search_r1` uses, so the threshold is not a tell for
which side of the pair a task is on.

Both bounds are calibrated on four measured runs (devspace 1107990,
`/user/user2/harbor-runs/`), best over steps `0..99`: repaired `0.3733`
and `0.3570`, defective `0.2237` and `0.1587` (the latter is the frozen
evidence), untrained step 0 about `0.045`–`0.051`. Thinnest margin is `0.056`,
on the defective side.

**The rerun budget is deliberately shorter than the 200-step frozen run.** The
defect is high-variance, and the variance grows with steps: the two measured
defective runs share identical code, args and retrieval server, yet one
collapsed to ~9-token responses by step 20 and plateaued at `0.1657` while the
other held ~650-token responses and was still climbing at `0.2713` when it was
killed at step 119. At step 99 both are fully observed, so the gap
`[0.2237, 0.3570]` is measured on both sides; at step 199 the defective side is
censored and no threshold there can be shown to reject it. Do not extend the
graded horizon to 199 without first running a defective 200-step run.

## Environment requirement

- Image: `slime-v0.3.2` (task.toml's `environment.image`), the upstream
  `slimerl/slime:v0.3.2-cu129` image pinned by digest — CUDA 12.9, sglang
  0.5.15.post1. There is no per-task Dockerfile.
- `environment/prepare.sh` installs `faiss-gpu-cuvs` into `$AUTORLBENCH_PREPARED`
  (not into the image, which the other slime tasks share) and
  exports it via `PYTHONPATH`. The `-cuvs` build is the CUDA 13 one, which is
  what this environment is; plain `faiss-gpu` is the CUDA 12 build.
- `environment/prepare.sh` then stages the five assets a rerun needs into
  `$AUTORLBENCH_WORKSPACE`: the `Qwen2.5-3B` checkpoint and its `torch_dist`
  conversion, the `e5-base-v2` query encoder, `nq_hotpotqa_train/{train,test}
  .parquet` (`autorlbench_dataset --recipe search_r1_nq_hotpotqa`) and
  `wiki-18/{e5_Flat.index,wiki-18.jsonl}` (`--recipe wiki18_e5`). Roughly 90 GB,
  cached under `$AUTORLBENCH_DATASETS`/`$AUTORLBENCH_MODELS` and hardlinked per run.
  The file stages the same assets with the same logic as the healthy twin
  (`tasks/slime_search_r1`); a change on one side must be mirrored on the other.
- `task.toml`'s `environment.env` deliberately omits `HF_HUB_OFFLINE` /
  `TRANSFORMERS_OFFLINE`: the framework sets them per stage
  (`_network_policy_env` in `autorlbench/runtimes/base.py`), whereas a task-level copy would also
  apply to `prepare` and disable the downloads above.
- Use the Docker agent backend.
- Issue diagnosis uses the frozen historical snapshot only. A solution rerun
  additionally needs the retrieval server on `:8000`. `prepare.sh` cannot hold
  it — the prepare stage is a container created and removed around the script
  (`autorlbench/runtimes/providers/docker.py::close_sandbox`) — so
  `run_qwen2.5_3B.sh` starts it itself when the port has no listener, and waits
  for readiness. `prepare.sh` pays the slow half of that start in advance by
  warming the corpus's `datasets` arrow cache under
  `$AUTORLBENCH_WORKSPACE/.hf_datasets_cache` (exported through `prepared/env` as
  `HF_DATASETS_CACHE`; the persistent copy lives in `$AUTORLBENCH_DATASETS`).
  A cold start still reads the 60 GiB index into GPU memory and takes minutes.
- Model weights and historical training checkpoints are not dataset assets.

The workspace code and the Search-R1 recipe are distributed under Apache-2.0.

The grader requires validation at the declared final step and rejects evaluation
points outside the declared horizon before applying the existing best-score gate.
A passing earlier point alone does not establish completion.
