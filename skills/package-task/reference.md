# Package task reference

## Required inputs

If any item is missing, stop.

### Task location

- Source workspace path
- Target directory: `tasks/<domain>/<task_id>/`
- `task_id`, title
- Model name, dataset name, framework, algorithm

`failure_reason` is one sentence written from the filtered diff, including the metric symptoms the user gave.

### Two commits

- git repo path
- buggy commit (the workspace matches this one)
- baseline commit (the expected correct state)
- Unrelated hunks to exclude (leave empty if none)

Attribution command:

```bash
git diff <buggy> <baseline>
```

Keep only code and launch scripts, plus the training settings, hyperparameters, rewards, and data processing they directly reference. Exclude:

- `logs/`, `wandb/`, `outputs/`, `*.log`
- checkpoint, `*.safetensors`, `*.bin`, `*.pt`, `*.ckpt`
- Train and validation data files (parquet, jsonl data, etc.)

### How this training run is launched

- Launch script path relative to `workspace/` (this is `training.script` in `task.toml`)
- Positional args (this is `training.script_args`) and required env vars
- In-container model path
- In-container train set path and validation set path
- Step count and eval interval (when fixed, write them into `solution/instruction.md`)
- Path of the failed run's logs in the source workspace
- Whether the source includes wandb; keep it if so

### Environment and assets

- Environment: an existing `docker/images/` name matching the recorded stack (see the table in `docker/README.md`), or the raw task's one-line digest-pinned Dockerfile, which becomes a new `docker/images/<name>.Dockerfile`
- GPU count to declare under `[environment] gpus` in `task.toml` (never device ids)
- Each HF asset: repo (`org/name`), `--type` (`model` or `dataset`), the pinned 40-char commit sha, `--dest`
- URL assets (e.g. pure-Python wheels the image lacks): URL, SHA-256, byte size
- Any derived asset the recorded run built from fetched data (parquet files, FAISS indexes, torch_dist weights) that `prepare.sh` must rebuild
- Any env vars the run needs beyond what the harness already sets (they go through `$AUTORLBENCH_PREPARED/env`)

Do not collect a build recipe: the repo keeps only one-line digest pins, and an elaborate stack is built and pushed by the contributor to a public registry (ghcr.io, Docker Hub, …).

### Scoring keys (needed only when the task gets a solution tier)

Collect these only after deciding the task has a solution tier (see the decision rule in [SKILL.md](SKILL.md#does-the-task-get-a-solution-tier)). A solution tier needs an eval-set key:

- Eval-set key name
- Where the key appears: text log, offline wandb, or both
- One real log sample containing the key and a number
- `final_step`
- `historical_final_baseline`
- `required_final_score`
- Optional: `score_mode` (`best` or `final`, default `best`), `required_final_score_max`

The scoring method is optional. If not given, the score depends entirely on the eval-set key.

When the user specifies a scoring method, each reference score needs a key name, threshold, weight, where it appears, and one log sample, plus a pass rule. A reference score can be a training score or another logged value such as learning rate. See [examples.md](examples.md) for a filled-in example.

### Historical evidence (solution tier only)

- The failed run's eval point series (step, value), used to write `historical_evidence_manifest.json` and calibrate the threshold

## Copy verbatim

After copying, `cmp` must match. Keep LF line endings.

| Target | Source |
| --- | --- |
| `evaluations/issue/instruction.md` | `tasks/reward/slime_retool_length_reward_bonus/evaluations/issue/instruction.md` |
| `evaluations/issue/verifier/judge_prompt.md` | `verifier/judge_prompt.md` in the same directory |
| `evaluations/issue/verifier/judge.py` | `verifier/judge.py` in the same directory |
| `evaluations/issue/verifier/test.sh` | `verifier/test.sh` in the same directory |
| `evaluations/solution/verifier/test.sh` | `tasks/reward/slime_retool_length_reward_bonus/evaluations/solution/verifier/test.sh` |

These five files in `environment/search_r1_zero_query_embedding` and `infrastructure/verl_moe_token_normalization` are byte-identical to it.

When needed, copy `environment/docker-compose.yaml` verbatim from `tasks/reward/slime_retool_length_reward_bonus/environment/docker-compose.yaml`.

## issue/oracle.md

```markdown
# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between `<buggy>` and `<baseline>`.

Expected verdict: **unhealthy**.

## Root cause

- `<path-or-symbol>`
  - kind: `<symbol|hparam|script>`
  - observed: `<buggy side>`
  - expected: `<baseline side>`

## Observable impact

- Primary source evaluation metric: `<metric_key>`.
- <symptoms from logs, not listed as separate root causes>

## Trigger conditions

- <trigger conditions from the diff>

## Documentation reconciliation

- <which log observations are symptoms>
```

With more than one Root cause, add at the end of that section:

```text
Listed causes are all essential. Identifying only part of them must receive partial root_cause_score.
```

## issue/verifier/README.md

A fixed description of how this tier is graded, written after `tasks/reward/slime_retool_length_reward_bonus/evaluations/issue/verifier/README.md`: the judge reads `$AUTORLBENCH_RESULTS/agent_answer.md` and compares it with the hidden `oracle.md`, weights 0.2 / 0.6 / 0.2, pass mark 0.8, requires `AUTORLBENCH_JUDGE_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, writes the result to `$AUTORLBENCH_RESULTS/result.json`, and writes the raw judge response plus the request that produced it to `$AUTORLBENCH_RESULTS/judge-raw.json`.

## issue/verifier/gold.yaml (optional)

For reviewers only; machine grading does not read it. If written, it must match `oracle.md`:

```yaml
category: <implementation|hyperparameter|reward>
framework: <framework>
task: <task_id>
verdict: <healthy|unhealthy>
root_cause:
  - <one root cause, matching Root cause in oracle.md>
evidence_metrics:
  - <one piece of observable evidence>
fix:
  must:
    - <required fix action>
```

## solution/instruction.md

The sections from here through `grader_spec.json` apply **only when the task gets a solution tier** (decision rule in [SKILL.md](SKILL.md#does-the-task-get-a-solution-tier)). An issue-only task produces none of these files.

The solution protocol is debug-then-report: the agent repairs the workspace and submits `report.md`, and the **harness** re-runs the experiment afterwards. So `instruction.md` never tells the agent to launch anything.

From the start through "Training reward alone does not establish held-out health.", reuse `tasks/reward/slime_retool_length_reward_bonus/evaluations/solution/instruction.md`. Keep its `## Your job` section verbatim too — the report gate is identical for every task. Then replace `## What the system will run` with this task's user input:

```markdown
## What the system will run

After you submit the report, the system deletes everything under
`$AUTORLBENCH_RESULTS/experiment_logs`, then re-runs the experiment in your
repaired workspace as a foreground process:

```bash
<the exact command from run_spec.json>
```

<with this environment: the env block from run_spec.json, when it has one>

The effective run uses:

- model `<in-container model path>`
- train data `<train set path>`
- validation data `<validation set path>`
- <fixed step count and eval interval, e.g. `--num-rollout 100`, `--eval-interval 10`>
- `WANDB_MODE=offline`

A task-specific grading script then reads **only this fresh run's** files under
`$AUTORLBENCH_RESULTS/experiment_logs` and scores the key metrics; logs written
anywhere else are ignored. Do not copy, edit, or fabricate log metrics.
Otherwise the run is scored 0.
```

Call out any trap that would make the rerun produce the wrong run, since the agent is the only one who can fix it beforehand: a budget baked into the launch script that the rerun does not override (`--num-rollout 3000` where the protocol wants 70), a launcher that builds its own Ray runtime-env JSON so `WANDB_DIR` never reaches the training process, a guard that refuses a run-name that already exists. State what the agent must change, not what the defect is.

## solution/verifier/resources/run_spec.json

The machine-readable half of the same protocol: what the harness runs, with what environment, for how long, and which log roots it clears first. Values may reference `$AUTORLBENCH_RESULTS` / `$AUTORLBENCH_WORKSPACE`; the harness expands them per backend. `env` must not declare `AUTORLBENCH_*` itself.

```json
{
  "schema_version": 1,
  "command": "<shell command, run from the workspace root, with every override the protocol requires>",
  "env": {"LOG_DIR": "$AUTORLBENCH_RESULTS/experiment_logs"},
  "timeout_s": 28800,
  "log_roots": ["experiment_logs"]
}
```

`timeout_s` is the run's real wall-clock budget — without it the generic evaluation timeout applies and cuts a multi-hour run off at the knees. Omit the file entirely for a solution task with nothing to re-run (a pure protocol repair): the report gate still applies, the rerun is simply skipped.

## solution/oracle/README.md

Fill in from the same filtered diff. The machine grading paragraph only repeats this task's user-given keys, thresholds, and weights.

```markdown
# Canonical solution oracle

## Root cause

- `<path-or-symbol>`
  - kind: `<kind>`
  - observed: `<buggy>`
  - expected: `<baseline>`

## Observable impact

- Primary source evaluation metric: `<metric_key>`.

## Trigger conditions

- <same as issue oracle>

## Required remediation

- `<path>`: revert to the baseline's `<expected>`.

Machine grading reads only the new experiment logs.
<state keys, thresholds, weights, and pass rule per grader_spec>
```

## solution/oracle/task.yaml

```yaml
task_id: <task_id>
category: <family or recipe category>
framework: <framework>
launch_script: <launch script relative to workspace/>
model: <model name>
dataset: <dataset name>
algo: <algo>
expected_verdict: <healthy|unhealthy>
evidence_kind: <historical_run|dirty_agent_failure>
buggy_commit: <sha>
baseline_commit: <sha>
eval_metric_key: <metric_key>
last_eval_step: <final_step>
causes:
  - path: <repo-relative path>
    kind: <symbol|hparam|script>
    observed: <buggy>
    expected: <baseline>
key_files:
  - <files touched by the diff>
external_assets_not_in_git:
  - <asset delivered by prepare.sh, one-line description>
```

When the same framework already has a more complete `task.yaml` (e.g. `infrastructure/verl_moe_token_normalization`), reuse that skeleton, but still write in every root cause from this diff.

## solution/oracle/standard_answer.txt

The judge's hard criteria, plain text, with a fixed header:

```text
STANDARD_ANSWER_V1
CASE_ID: <task_id>
BENCHMARK_GROUP: <B-...>
EXPECTED_VERDICT: <healthy|unhealthy>
EXPECTED_ATTRIBUTION: <one sentence, the root cause the diff points to; for healthy, write none and the reason>
PRIMARY_METRIC: <metric_key>

REQUIRED_ALL:
- <each item that must be named; list each root cause separately>

GLOBAL_JUDGE_POLICY:
- The score must be exactly 0 or 1.
- Agent A must explicitly classify the run as healthy or unhealthy.
- Every REQUIRED_ALL item must be satisfied.
- A verdict mismatch forces score 0.
- Equivalent technical wording in any language is acceptable.
- task.yaml may clarify targets and aliases but may not relax REQUIRED_ALL.
```

`healthy` tasks add one more rule: reporting a benign but unusual setting as a defect scores 0.

## solution/oracle/historical_evidence_manifest.json

The failed run's metric series, which reviewers use to recheck the threshold:

```json
{
  "schema_version": 1,
  "evidence_kind": "dirty_agent_failure",
  "task_id": "<task_id>",
  "family": "<family>",
  "metrics": {
    "metric_status": "log_reconstructed",
    "expected_metric_key": "<metric_key>",
    "actual_metric_key": "<metric_key>",
    "eval_points": [
      { "step": 0, "eval_score": 0.0307, "eval_key": "<metric_key>", "truncate_ratio": null }
    ]
  }
}
```

## grader_spec.json

```json
{
  "schema_version": 2,
  "task_id": "<task_id>",
  "metric_key": "<user-supplied eval key>",
  "final_step": 0,
  "historical_final_baseline": "<user-supplied value>",
  "required_final_score": "<user-supplied value>",
  "log_roots": ["experiment_logs"]
}
```

Optional keys: `score_mode` (`best` or `final`; `grade_submission.py` defaults to `best`), `required_final_score_max` (an upper bound that rejects absurd fabricated values). Write all numbers as strings; they are read with `Decimal`.

If no scoring method is specified, do not add other score keys. When the user specifies reference scores, add them to the same `grader_spec.json` as a **flat triple**; do not use `components` / `pass_rule`:

```json
{
  "training_reward_key": "<reference score key>",
  "required_training_reward": "<threshold>",
  "validation_weight": "0.8",
  "training_weight": "0.2"
}
```

Common naming sets: `training_reward_key` / `required_training_reward` / `training_weight`, and `learning_rate_key` / `required_learning_rate` / `learning_rate_weight`. The eval-set side is always `validation_weight`. Name keys after what the reference score is, and list them all in the required-keys list in `grade_submission.py`.

`grade_submission.py` reads the sibling `../resources/grader_spec.json`. Self-test with the sample string: it must extract the user-given key and the number in the sample. Outputs:

- `experiment_metrics.json`
- `metrics.json`
- `result.json`: `{"passed": <bool>, "score": <number>, "metrics": {...}}`

`resources/README.md` states that the entry point is `test.sh`, that scores come only from `$AUTORLBENCH_RESULTS/experiment_logs`, and lists the same keys and thresholds as the spec. `verifier/README.md` adds one more layer: which space the key reads (score or accuracy), why `score_mode` was chosen, which points on the historical curve the threshold was calibrated against, and the thinnest margin.

## Environment image

`environment.image` in `task.toml` is the `<name>` of some `<name>.Dockerfile` under `docker/images/`. Every runtime Dockerfile is a **single digest-pinned `FROM` line** — either an upstream framework image or an image the contributor built once and pushed to a public registry. The reference is resolved from that `FROM` line by `EnvironmentImage` (an optional `# autorlbench-image:` header still wins, for backwards compatibility) and reaches the Docker backend, `build-images`, and the Harbor adapter unchanged. Nothing is built at evaluation time: `python3 -m autorlbench build-images <name>` just pulls the digest on demand.

Images are shared across tasks, and their contents are those tasks' common ground-truth environment. Do not modify an existing Dockerfile for a new task; that would also change the other tasks' environments. If no existing image fits, add a new `docker/images/<name>.Dockerfile` containing the raw task's one-line pin, named per the slot grammar in `docker/README.md` (only what is actually baked into the image: framework, engine, torch, flash-attn, CUDA, python).

The framework (verl / slime) usually travels with the task's workspace checkout rather than the image; environments that bake it in lead their name with its version (e.g. `slime-v0.3.2-...`).

## environment/prepare.sh

```bash
#!/usr/bin/env bash
# <task_id>: prepare stage.
#
# Runs in a separate prepare-stage container: root, the `<image-name>` image,
# **the only networked stage**, and $AUTORLBENCH_WORKSPACE is writable.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"

# --- Base model ---------------------------------------------------------------
# <why dest must be exactly this path: which guard on which line of the launch script looks for it>
# Pinned revision, for reference: <org/repo> @ <40-char commit sha>
python3 "$env_dir/fetch_hf.py" \
    --repo <org/repo> \
    --dest "$W/<relative model path>"

# --- Dataset ------------------------------------------------------------------
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo <org/repo> \
    --dest "$W/<relative dataset path>"
```

`fetch_hf.py` lives in the task's own `environment/` directory — copy it verbatim from an existing task (most tasks carry the same file) and invoke it with the image's own interpreter. It downloads whole HF repos into the shared caches, keyed by repo id, and hard-links the bytes into `--dest`, so tens of GB of weights are not duplicated on disk.

Fetches are cache-first **without checksum verification**, so the pinning is yours to write:

- for each HF repo, record the pinned 40-char commit sha in a comment next to the call;
- for anything fetched by URL (e.g. pure-Python wheels the image lacks), verify SHA-256 and byte size in-flight and fail loudly on drift;
- guard derived outputs (parquet files, indexes) with a `.complete` marker so re-runs skip finished work.

Use `--cache-as <owner/name>` when a repo's name would reveal the task's answer: only the cache directory name changes, the bytes stay shared.

Available variables:

| Name | What it is |
| --- | --- |
| `$AUTORLBENCH_WORKSPACE` | The workspace the agent receives, writable during prepare; assets go here |
| `$AUTORLBENCH_PREPARED` | Output directory; writing `KEY=VALUE` lines to `$AUTORLBENCH_PREPARED/env` passes env vars to the agent/verifier (read line by line as data, never sourced) |
| `$AUTORLBENCH_MODELS` / `$AUTORLBENCH_DATASETS` | Download caches shared across runs, read-only in the agent/verifier stages; **task scripts must not reference them directly**, since the harbor backend deletes them after prepare |
| `$AUTORLBENCH_ENVIRONMENT` | The task's own `environment/` (read-only) |

When prepare needs a format conversion (e.g. HF → Megatron torch_dist), probe for CUDA first and, if unavailable, **skip instead of erroring**. The harbor backend runs prepare as a build-time layer with no GPU, so the training script's own guard does it in the agent stage.

## task.toml

```toml
title = "<title on the website task card>"
license = "MIT"
created_date = "<YYYY-MM-DD>"
domain = "<configuration|reward|data|environment|algorithm|infrastructure>"
tags = ["<framework>", "<algo>", "<tag>"]

[submitter]
name = "<name>"

[model]
name = "<model name>"

[dataset]
name = "<dataset name>"

[training]
framework = "<framework>"
algo = "<algo>"
script = "<launch script path relative to workspace/>"
script_args = ["<arg>"]          # only when the launch script takes positional args

[environment]
image = "<environment name under docker/images/>"
# Optional GPU count; the docker backend turns it into `--gpus <n> --shm-size=16g`,
# harbor synthesizes the compose device reservation from it. Never device ids.
gpus = 8

[environment.env]                # optional, run-time env vars the task needs
VLLM_ATTENTION_BACKEND = "XFORMERS"

[outcome]
failure_type = "<see below>"
failure_reason = "<one sentence from the filtered diff, with the user-given metric symptoms>"
```

`failure_type` only needs to be non-empty. Values used by existing tasks: `reward_hacking`, `evaluation_under_sampling`,
`data_difficulty_mismatch`, `retriever_query_device_copy`, `training_collapse`, `implementation_bug`.

The validator rejects:

- Missing `image`, or `image` with no matching `<name>.Dockerfile` under `docker/images/`
- Legacy fields `conda` / `requirements` / `dockerfile`
- `script` that is absolute, contains `..`, or points to a file not in `workspace/`
- `script_args` without `script`; `script_args` that is not a list of strings
- `env` values that are not strings (`CUDA_VISIBLE_DEVICES = 0` is rejected), or that set `AUTORLBENCH_*`
- `gpus` that is not a non-negative integer

Declaring `framework` without `script` only gives a warning, but the task card relies on it to record how the task runs, so do not omit it.

## README.md

Write it per [readme-template.md](readme-template.md).
