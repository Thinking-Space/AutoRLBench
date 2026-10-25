---
name: package-task
description: >-
  Packages a buggy RL training workspace (a raw task, a git commit pair, or a
  historical snapshot) into an AutoRLBench task under tasks/<domain>/<task_id>/:
  task.toml, README.md, workspace/, environment/prepare.sh, and issue/solution
  evaluations. Use when the user asks to package, build, or convert a run into
  the task format, for any model, framework, or dataset.
---

# package-task

Packages one faulty RL training run into `tasks/<domain>/<task_id>/`. `<domain>` is one of
`configuration`, `reward`, `data`, `environment`, `algorithm`, `infrastructure`.

| File | Contents |
| --- | --- |
| [SKILL.md](SKILL.md) | Workflow and rules (this file) |
| [reference.md](reference.md) | Template for every output file |
| [examples.md](examples.md) | A filled-in user input |
| [readme-template.md](readme-template.md) | Template and style guide for the task `README.md` |

Authoritative sources in the repo: `autorlbench/tasks/validator.py` (validation rules), `docker/README.md` (environment images).
For finished tasks, see `tasks/reward/slime_retool_length_reward_bonus/` (issue + solution) and
`tasks/data/verl_taco_difficulty_mismatch/` (issue only).

## Outputs

```text
tasks/<domain>/<task_id>/
├── README.md                     # for reviewers, not visible to the agent
├── task.toml                     # task card: metadata + environment
├── workspace/                    # defective side's code, launch script, and frozen evidence
├── environment/
│   ├── prepare.sh                # the only networked stage: fetch model, fetch data
│   ├── docker-compose.yaml       # optional: multi-GPU / large shm on Harbor
│   ├── data_manifest.json        # optional: when data ships with the task
│   └── source_manifest.json      # optional: when packaging modified the source
└── evaluations/
    ├── issue/                    # diagnosis
    │   ├── instruction.md
    │   ├── oracle.md
    │   └── verifier/{test.sh,judge.py,judge_prompt.md}
    └── solution/                 # optional: fix, then the harness reruns
        ├── instruction.md
        ├── oracle/
        └── verifier/{test.sh,tests/grade_submission.py,resources/{grader_spec.json,run_spec.json}}
```

Do not produce `agent_trace.json`, `environment/Dockerfile`, `requirements.txt`, or `validation/`.
Tasks do not ship their own image: `environment.image` in `task.toml` points to `docker/images/<name>.Dockerfile`.

## Required inputs

If any item is missing, stop and ask. Field list in [reference.md](reference.md#required-inputs), example in [examples.md](examples.md).

- Source (one of three forms, see next section)
- `domain`, `task_id`, title; model, dataset, framework, algorithm
- Launch script (relative to `workspace/`), positional args, env vars; in-container paths for model, train set, and validation set
- Logs of the failed run (keep W&B too if present)
- Environment image name, GPUs to use
- For each asset: source, 40-char revision, SHA-256 and byte size, destination
- Solution tier: metric key, one log sample, `final_step`, `required_final_score`, and **how many runs per side** the threshold was calibrated on

## Three source forms

| Form | Where the root cause comes from | Typical |
| --- | --- | --- |
| 1. Two git commits | `git diff <buggy> <baseline>` | Injected defect |
| 2. raw task: `buggy_workspace/` + `fixed_workspace/` | `diff -rq`, then `diff -u` per file | See [CONTRIBUTING.md](../../CONTRIBUTING.md) for format |
| 3. Historical snapshot, no control | The user-supplied `bug_root_cause`; if missing, ask. Do not infer it from logs yourself | `data/verl_taco_difficulty_mismatch` |

The root cause comes only from code, launch scripts, and the hyperparameters, rewards, and data processing they directly reference.
Differences in logs, W&B, checkpoints, weights, and data files never count as root cause.

Form 2 **publishes only the defective side**. Putting both sides in `workspace/` hands over the answer with a single `diff -r`:

| Item in the raw task | Destination |
| --- | --- |
| Defective-side code + launch script + run artifacts | `workspace/` |
| Changed files from the control side | `evaluations/solution/oracle/` |
| Control-side run results and curves | `evaluations/solution/oracle/` (used to calibrate the threshold) |
| The raw `Dockerfile` | Map to an existing image under `docker/images/`, or add a new one |
| The raw `README.md` | Rewrite as the task `README.md` |

## Workflow

1. **Identify the root cause.** Take the diff for the source form and filter out unrelated parts. If nothing substantive remains after filtering, stop.
2. **Copy the evaluation scaffolding.** Copy issue `instruction.md`, `judge.py`, `judge_prompt.md`, `test.sh`,
   and solution `verifier/test.sh` verbatim from `tasks/reward/slime_retool_length_reward_bonus/`,
   then check them with `cmp`.
3. **Write the oracle.** Use the diff from step 1 to write `evaluations/issue/oracle.md` and `solution/oracle/`.
   When there are multiple root causes, `oracle.md` must state that naming only some of them earns partial credit.
4. **Write the solution tier** (only when there is a control side or a post-fix run): `instruction.md`, `grader_spec.json`,
   `run_spec.json`, `grade_submission.py`. If the log format matches, reuse an existing `grade_submission.py` and change only the spec.
   `instruction.md` tells the agent to repair and submit `report.md` and **not** to launch training; `run_spec.json` is what the
   harness runs on its behalf. The two must agree — the agent is graded on a run it never starts, so a command or budget that
   appears in only one of them is how a correct repair still scores 0.
5. **Assemble the workspace.** Add the defective-side code and frozen evidence, slim and sanitize it (see below), then write `environment/prepare.sh`,
   `task.toml`, and `README.md`.
6. **Offline self-check.** Replay the grader on the log samples. With a control side, replay once per side: the defective side must fail and the control side must pass.
7. **Validate.** `python3 -m autorlbench validate tasks/<domain>/<task_id>` must report no errors.
   On a machine with Docker and network, also run `./docker/warm-cache.sh <task_id>` to confirm `prepare.sh` runs end to end.

## Rules

### Issue attribution

- For each root cause, state the path or symbol, observed (defective side), and expected (control side).
- Logs and curves go only in Observable impact, never as root cause.
- Judge weights are fixed at verdict / root cause / evidence = 0.2 / 0.6 / 0.2, pass mark 0.8.
- **The defective arm also improved** is a common trap: the root cause is "lower than control", not "no learning". List
  "training did not improve" as a wrong attribution in the oracle.

### Threshold

- The threshold sits between the defective side's upper bound and the control side's lower bound. Report the margin to both.
- The threshold must be above the control side's value at step 0; otherwise fixing the code without finishing the run would pass.
- `required_final_score_max` is only for rejecting fabricated values (accuracy caps at `1.0`).
- State **how many runs per side** it was calibrated on. With one run per side, report margins only, not variance, and say so in the README.
- `final_step` is the last step of the **rerun**, not the last step of the frozen evidence.

### Frozen evidence: slimming and sanitizing

| Keep | Drop |
| --- | --- |
| Training logs, launch logs, server logs, `experiment.json` | Most per-step rollout dumps (keep three samples: first, middle, last) |
| All per-step metric files (a few KB each) | W&B `*.wandb` binaries, `latest-run/` |
| Offline W&B `files/`, `logs/` | Checkpoints, weights, rebuildable derived assets |

Sanitizing changes only labels and paths, **never any metric value**. Remove: the words `broken` / `fixed`, absolute paths,
usernames, hostnames, IPs, and any port or GPU index that reveals which side it is. Record every replacement in
`environment/source_manifest.json` with the sha256 before and after. After sanitizing, scan again:

```bash
grep -rIl 'broken\|fixed\|<original path fragment>\|<original username>' workspace/
```

Any remaining hit must be an upstream word, with the reason noted in the manifest.

### Derived assets

Anything computable from shipped content plus a pinned model (FAISS indexes, torch_dist weights) is generated by
`environment/<script>.py` during prepare, not shipped. The generator must be self-contained, fall back to CPU,
and assert only **the side unrelated to the defect**: `environment/` is readable by the agent, so asserting query-side quality here
points at the defect. Guard outputs with a `.complete` marker, hard-link them into the workspace, and do not write `--expect`.

### Environment

- Pick an image from `docker/images/` that matches the recorded stack. Do not modify existing Dockerfiles. If none fits, add one
  as described in `docker/README.md`; if it is not built yet, write `not yet built` in the README.
- `prepare.sh` fetches assets with `autorlbench_fetch_hf` / `autorlbench_fetch_url` / `autorlbench_dataset`,
  each with `--expect` and `--expect-size`. HF revisions are 40-char commit shas.
- Set `--dest` to `$AUTORLBENCH_WORKSPACE/<path the launch script actually reads>`. `$AUTORLBENCH_MODELS` /
  `$AUTORLBENCH_DATASETS` are only caches; task scripts must not reference them directly.
- GPUs go in `[environment.env]` of `task.toml`, and values must be strings. Do not set `AUTORLBENCH_*`
  or `HF_HUB_OFFLINE` here: the framework sets online and offline mode per stage.

### task.toml and README.md

- `task.toml` fields are in [reference.md](reference.md#tasktoml). The file `training.script` points to must exist in
  `workspace/`.
- Write `README.md` per [readme-template.md](readme-template.md): a one-sentence summary, an identity table, and the
  Difficulty / Reference solution / Verification sections. The agent sees `instruction.md`;
  the README is for reviewers only and must not be mounted into the agent stage.

## Pre-completion checklist

- [ ] Skeleton complete; no `agent_trace.json`, `environment/Dockerfile`, `validation/`
- [ ] The four issue files and solution `test.sh` all match their source under `cmp`
- [ ] Oracle root causes match the filtered diff; partial-credit sentence present for multiple causes
- [ ] Grader replay: defective side exit 1, control side exit 0
- [ ] Solution tier: `instruction.md` forbids launching training and requires `report.md`; its command, env and budget match `run_spec.json`
- [ ] Workspace has the launch script and failure logs; sanitize scan shows only upstream hits
- [ ] All `env` values in `task.toml` are strings, no `AUTORLBENCH_*`
- [ ] Every asset in `prepare.sh` has `--expect` / `--expect-size`, and every revision is a 40-char sha
- [ ] `python3 -m autorlbench validate <task_dir>` reports no errors
- [ ] README states how many runs per side calibrated the threshold, and which steps were not verified
