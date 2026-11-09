# Contributing a task

You contribute the raw material: a real training run that went wrong, and optionally the run where you fixed it. An agent turns that into a task with the
[package-task](skills/package-task/SKILL.md) skill. 

## What to prepare

We suggest something like this:

```text
<your_task>/
├── buggy_workspace/   code that reproduces the defect, plus that run's artifacts
├── fixed_workspace/   the same tree with the fix, plus its run's artifacts   (optional)
├── Dockerfile         names the environment both runs used (a digest-pinned FROM line
│                      is the cleanest form)
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
- **A Dockerfile that pins the environment** It's suggested that you package the heavy
training stack — torch, CUDA, compiled kernels, the framework — into a single image,
build it and push to a public registry (ghcr.io, Docker Hub, …), or reuse an existing
environment in [docker/images/](docker/images/). The cleanest pin is a single
digest-pinned line:
  ```dockerfile
  FROM ghcr.io/<you>/<your-image>@sha256:4c43bbf1...
  ```

  The digest is what makes the environment deterministic. Model and dataset
  downloads and other small environment touches can reside in separate scripts: a couple
  of pure-Python wheels the image lacks, env vars the run needs, or specific
  post-processing of the dataset or model.
- **The README** states the defect, the fix as a diff, the measured outcome of both arms,
and every plausible-but-wrong diagnosis. If you ran each arm once, say so — a
single-seed result supports margins, not variance.
- **Leave out** model weights, checkpoints, and anything derivable from what you did
ship. Record how to obtain them in the scripts instead.

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
├── Dockerfile             CUDA 12.1, Python 3.10, torch 2.4.0, vLLM 0.6.3,
│                          flash-attn 2.6.3, faiss-cpu 1.14.2
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

It produces `tasks/<domain>/<task_id>/` in the [task layout](README.md#task-layout), and
if your environment is a new image, its one-line Dockerfile lands under
[docker/images/](docker/images/).

```bash
python3 -m autorlbench validate tasks/<domain>/<task_id>
```

It must report zero errors.

## Open a pull request

The generated task is already inside your clone, so branch, commit, and push it to your
fork, then open the PR against `main`. The packaging skill also emits a PR body
derived from the finished task README — paste it into the PR description (you may adjust
details, but keep the base format). Include the raw material only if it is
small enough to be useful as evidence.

Before opening the PR, you can use the repo's `rerun` command to check that the buggy run
actually runs in the packaged task — it runs the task's `prepare.sh`, then launches the
experiment:

```bash
python3 -m autorlbench rerun tasks/<domain>/<task_id>
```

The experiment does not have to run to completion: once training has produced a step, the
environment is verified and you can stop the run. If you ran the check, say so in one
sentence after the pasted PR body.
