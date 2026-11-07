---
name: package-task
description: >-
  Packages a buggy RL training workspace (a raw task, a git commit pair, or a
  historical snapshot) into an AutoRLBench task under tasks/<domain>/<task_id>/:
  task.toml, README.md, workspace/, environment/prepare.sh, and evaluations —
  an issue tier always, a solution tier only when the fix outcome is gradeable
  (decided during packaging, never assumed). Use when the user asks to package,
  build, or convert a run into the task format, for any model, framework, or
  dataset.
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
| [pr-body-template.md](pr-body-template.md) | Format of the PR body derived from the finished README |

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
│   ├── docker-compose.yaml       # exceptional: only for contracts the harness cannot express
│   ├── fetch_hf.py               # the HF fetch helper, copied verbatim from an existing task
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
A task whose stack matches no existing environment adds one new file there — a one-line digest-pinned `FROM` — instead.

## Required inputs

If any item is missing, stop and ask. Field list in [reference.md](reference.md#required-inputs), example in [examples.md](examples.md).

- Source (one of three forms, see next section)
- `domain`, `task_id`, title; model, dataset, framework, algorithm
- Launch script (relative to `workspace/`), positional args, env vars; in-container paths for model, train set, and validation set
- Logs of the failed run (keep W&B too if present)
- Environment: an existing `docker/images/` name matching the recorded stack, or the raw task's one-line
  digest-pinned Dockerfile (becomes a new environment); the GPU count to declare
- For each asset: source (HF repo or URL), the 40-char revision or wheel SHA-256 + byte size to pin, destination
- Only when you will build a solution tier (see [Does the task get a solution tier?](#does-the-task-get-a-solution-tier)):
  metric key, one log sample, `final_step`, `required_final_score`, and **how many runs per side** the threshold was
  calibrated on

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
| The raw `Dockerfile` | Map to an existing image under `docker/images/`, or add its one-line digest-pinned `FROM` as a new `docker/images/<name>.Dockerfile` |
| The raw `README.md` | Rewrite as the task `README.md` |

## Does the task get a solution tier?

**You decide this during packaging — it is never an obligation, and never assumed.** A control side (a fixed run, a
baseline commit) is an invitation, not a requirement: not every defect has a canonical fix, and not every fix has an
outcome a rerun can grade. Write a solution tier only when **all** of these hold:

- There is a control side or a post-fix run, and the comparison is clean — same seed, data, and budget; nothing but the
  fix differs (placement differences like GPU or port do not count, but say so).
- A metric the rerun's grader can read separates the arms with margin on both sides: a threshold fits between the
  defective side's upper bound and the control side's lower bound, and sits above the control side's step-0 value.
- **Any correct repair reaches that outcome, not just the recorded one.** When several defensible fixes produce genuinely
  different outcomes, or the fix's effect is qualitative, or a single-seed margin is too thin to trust, a threshold would
  grade repairs against one arbitrarily chosen fix — package issue-only instead.

When you decide against it, do not ask the user for scoring keys, do not write `evaluations/solution/`, and state the
reason in one line of the task README (e.g. "issue-only: the post-fix margin is single-seed and under one point").

## Workflow

1. **Identify the root cause.** Take the diff for the source form and filter out unrelated parts. If nothing substantive remains after filtering, stop.
2. **Decide the tiers.** Apply [Does the task get a solution tier?](#does-the-task-get-a-solution-tier). The issue tier is
   always written; the solution tier only when the decision rule says so.
3. **Copy the evaluation scaffolding.** Copy issue `instruction.md`, `judge.py`, `judge_prompt.md`, `test.sh`,
   and — when there is a solution tier — solution `verifier/test.sh` verbatim from
   `tasks/reward/slime_retool_length_reward_bonus/`, then check them with `cmp`.
4. **Write the oracle.** Use the diff from step 1 to write `evaluations/issue/oracle.md` and `solution/oracle/` (when there
   is a solution tier). When there are multiple root causes, `oracle.md` must state that naming only some of them earns partial credit.
5. **Write the solution tier** (skip entirely when step 2 decided against it): `instruction.md`, `grader_spec.json`,
   `run_spec.json`, `grade_submission.py`. If the log format matches, reuse an existing `grade_submission.py` and change only the spec.
   `instruction.md` tells the agent to repair and submit `report.md` and **not** to launch training; `run_spec.json` is what the
   harness runs on its behalf. The two must agree — the agent is graded on a run it never starts, so a command or budget that
   appears in only one of them is how a correct repair still scores 0.
6. **Assemble the workspace.** Add the defective-side code and frozen evidence, slim and sanitize it (see below), copy
   `fetch_hf.py` from an existing task's `environment/`, then write `environment/prepare.sh`, `task.toml`, and `README.md`.
7. **Offline self-check.** With a solution tier, replay the grader on the log samples once per side: the defective side must
   fail and the control side must pass.
8. **Validate.** `python3 -m autorlbench validate tasks/<domain>/<task_id>` must report no errors.
9. **Emit the PR body.** Compose it per [pr-body-template.md](pr-body-template.md) from the finished README — task path,
   summary sentence, identity table, evidence numbers — and present it to the
   user to paste into the PR. Do not write it into the task directory.

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
points at the defect. Guard outputs with a `.complete` marker and hard-link them into the workspace; if the output
is not byte-for-byte reproducible, do not pin it with a checksum.

### Environment

- Pick an image from `docker/images/` that matches the recorded stack; do not modify existing Dockerfiles. If none fits,
  the raw task's Dockerfile must be a single digest-pinned `FROM` on a publicly pullable registry (ghcr.io, Docker Hub, …):
  copy that line to `docker/images/<name>.Dockerfile`, named per the conventions in `docker/README.md`, and reference the
  name in `environment.image`. An elaborate stack is built and pushed once by the contributor; the repo keeps only the pin.
  If the image is not pushed yet, say so in the README (see the build-recipe exceptions in `docker/README.md`).
- `prepare.sh` is the only networked stage. Invoke the task-local `fetch_hf.py` with the image's own interpreter; it
  downloads whole HF repos into the shared caches and hard-links the bytes into the workspace. Fetches are cache-first
  without checksum verification, so record each pinned 40-char revision in a comment next to the call, and for anything
  fetched by URL (e.g. pure-Python wheels the image lacks) verify SHA-256 and byte size in-flight, failing loudly on drift.
  Use `--cache-as` when a repo name would reveal the task's answer.
- Set `--dest` to `$AUTORLBENCH_WORKSPACE/<path the launch script actually reads>`. `$AUTORLBENCH_MODELS` /
  `$AUTORLBENCH_DATASETS` are only caches; task scripts must not reference them directly.
- GPUs are a count — `gpus = <n>` under `[environment]` in `task.toml` — never device ids; the backend derives the
  device spec. Values under `[environment.env]` must be strings, and must not set `AUTORLBENCH_*` or `HF_HUB_OFFLINE`
  there: the framework sets online and offline mode per stage.

### task.toml and README.md

- `task.toml` fields are in [reference.md](reference.md#tasktoml). The file `training.script` points to must exist in
  `workspace/`.
- Write `README.md` per [readme-template.md](readme-template.md): a one-sentence summary, an eight-row identity table
  (Author / Affiliation / Type / Domain / Tags / Framework / Run time / Resources — `Run time` is the buggy run's
  duration), then the Difficulty / Reference solution / Verification sections, whose content is free-form except that
  the latter two need `- **issue**` bullets and, with a solution tier, `- **solution**` bullets. The agent sees
  `instruction.md`; the README is for reviewers only and must not be mounted into the agent stage.

## Pre-completion checklist

- [ ] Skeleton complete; no `agent_trace.json`, `environment/Dockerfile`, `validation/`
- [ ] The tier decision is recorded: issue-only tasks say why in the README, one line
- [ ] The four issue files (and solution `test.sh`, when present) all match their source under `cmp`
- [ ] Oracle root causes match the filtered diff; partial-credit sentence present for multiple causes
- [ ] With a solution tier: grader replay — defective side exit 1, control side exit 0
- [ ] With a solution tier: `instruction.md` forbids launching training and requires `report.md`; its command, env and budget match `run_spec.json`
- [ ] Workspace has the launch script and failure logs; sanitize scan shows only upstream hits
- [ ] `task.toml` declares an integer `gpus` count (no device ids); `env` values are strings, no `AUTORLBENCH_*`
- [ ] A new environment, if any, is one digest-pinned `FROM` line under `docker/images/`, named per `docker/README.md`
- [ ] Every fetched asset has a pinned 40-char revision or verified wheel sha256 + byte size, and a `--dest` the launch script reads
- [ ] `python3 -m autorlbench validate <task_dir>` reports no errors
- [ ] The PR body was emitted per [pr-body-template.md](pr-body-template.md) from the finished README
- [ ] With a solution tier: README states how many runs per side calibrated the threshold, and which steps were not verified
