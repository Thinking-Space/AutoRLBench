# Contributing a task

You contribute the raw material: a real training run that went wrong, and optionally the run where you fixed it. An agent turns that into a task with the
[package-task](skills/package-task/SKILL.md) skill. 

## What to prepare

We suggest something like this:

```text
<your_task>/
├── buggy_workspace/   code that reproduces the defect, plus that run's artifacts
├── fixed_workspace/   the same tree with the fix, plus its run's artifacts   (optional)
└── README.md          the defect, the fix, and the measured outcome
```

Nothing here has to be tidy, the packaging step sanitizes them. But we suggest:

- **The two workspaces differ only by the fix**, and each is self-contained: its own code,
its own inputs. Anything else that differs confounds the comparison the threshold is
calibrated on.
- **Each workspace carries its run's artifacts** where the run wrote them: full
stdout/stderr, the offline W&B directory (including `run-*.wandb`, or `wandb sync`
cannot replay the run), every per-step metric file, and a couple of rollout dumps as
samples.
- **A buggy run executed in one of the eight shared environments.** Pick the row in
[ENVIRONMENTS.md](ENVIRONMENTS.md) whose stack is closest to your run, **run the buggy run
inside that image**, and ship that run's artifacts — the recorded evidence must come from
the environment the task declares. Framework or dependency versions the image lacks, and
other small environment touches, reside in `environment/prepare.sh` rather than a new image.
There is no custom-image exception.
- **The README** states the defect, the fix as a diff, the measured outcome of both arms,
and every plausible-but-wrong diagnosis. If you ran each arm once, say so — a
single-seed result supports margins, not variance.
- **Leave out** model weights, checkpoints, and anything derivable from what you did
ship. `environment/prepare.sh` must download or rebuild them, along with every other
runtime environment addition the chosen shared image lacks.

Without a fixed run, the task is issue-only (diagnosis). A fixed run does not guarantee a
solution level: not every defect has a canonical fix or a reproducible post-fix outcome, and
the packaging step decides whether grading a repair is meaningful.

### A worked example

[`tasks/environment/search_r1_zero_query_embedding`](tasks/environment/search_r1_zero_query_embedding/)
started life as raw material in this form:

```text
search-r1-zero-query-embedding/
├── buggy_workspace/       Search-R1 + veRL, data/, run.py, serve_retriever.py,
│                          results/baseline32/{train.log, launch.log,
│                          retrieval_server.log, retrieval.jsonl,
│                          experiment.json, rollouts/, wandb/}
├── fixed_workspace/       the same tree with four lines changed in
│                          search_r1/search/retrieval_server.py,
│                          plus its own results/baseline32/
├── (environment)          the run was re-run in the vllm0.6.3-torch2.4.0-
│                          flashattn2.6.3-cu121-py3.10 shared image
├── prepare.sh             downloads the two models (Qwen2.5-1.5B-Instruct,
│                          e5-base-v2), rebuilds the FAISS index
└── README.md
```

Its README described the defect (a
cross-device `torch.as_tensor` copy that silently returns zero embeddings, so FAISS
retrieves a constant top-3 while training still improves), the fix as a four-line diff,
the measured outcome —

| Step | buggy `val/em` | fixed `val/em` |
| ---: | ---: | ---: |
| 0 | 3.12% | 13.28% |
| 20 | 11.72% | 31.25% |
| **32** | **15.62%** | **32.81%** |

— and why the obvious diagnoses (checkpoint, reward, data, model size, "training is not
working") are wrong: both arms improve, and the gap is already open at step 0. The
packaged task keeps one workspace, the fix sealed into the solution oracle, and the run
artifacts thinned and sanitized.

## Package it

First, fork and clone this repository, and run the packaging from the repo root — the
skill writes the task into `tasks/`, copies its evaluation scaffolding from an existing
task, and validates with the harness shipped here, so it needs a local checkout. Keep
the raw material outside `tasks/`; anywhere else is fine.

```bash
git clone https://github.com/<you>/AutoRLBench.git
cd AutoRLBench
```

Then point an agent at the raw material with the
[package-task](skills/package-task/SKILL.md) skill shipped in this repo:

```text
Package <path-to-your_task> as an AutoRLBench task using the package-task skill at skills/package-task in this repository.
```

It produces `tasks/<type>/<task_id>/` in the [task layout](README.md#task-layout),
with the environment you ran the buggy run in recorded — digest-pinned — in the
task's `task.toml` (see [ENVIRONMENTS.md](ENVIRONMENTS.md)). The packaging skill does not
stop at schema validation: ask an agent to review the finished task with the
[validate-task](skills/validate-task/SKILL.md) skill before opening the PR:

```text
Validate tasks/<type>/<task_id> using the validate-task skill at skills/validate-task in this repository.
```

## Open a pull request

The generated task is already inside your clone, so branch, commit, and push it to your
fork, then open the PR against `main`. The packaging skill also emits a PR body
derived from the finished task README — paste it into the PR description (you may adjust
details, but keep the base format). Include the raw material only if it is
small enough to be useful as evidence.

Before opening the PR, use the repo's `rerun` command to check that the shared image and
the task's `prepare.sh` can start the experiment. The framework pulls the digest-pinned
shared image, launches prepare and rerun containers from it, runs `prepare.sh`, and then
starts the experiment declared by `run_spec.json`:

```bash
python3 -m autorlbench rerun tasks/<type>/<task_id>
```

The experiment does not have to run to completion: a few training steps are enough. Watch
`outputs/rerun/<task_id>/<timestamp>/rerun.log`, and stop the run after it has produced
those steps. The persistent output also contains `command.sh`, `prepare.log`, `run.log`,
the experiment's `results/`, and the salvaged workspace copy; a run that finishes
naturally also writes `report.json`. The
`validate-task` review reads these logs and treats a rerun as successful only when the
experiment reached a few steps without failing on image, dependency, model, or dataset
setup. If you interrupted the startup check after those steps, say so in one sentence
after the pasted PR body.
