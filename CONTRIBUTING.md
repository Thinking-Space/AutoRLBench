# Contributing a task

You contribute the raw material: a real training run that went wrong, and when you have
one, the run where you fixed it. An agent turns that into a task. This file describes what
the raw material must contain; the [package-task](skills/package-task/SKILL.md) skill
describes how it is packaged.

The site version of this guide is at
[thinking-space.github.io/AutoRLBench/contribute.html](https://thinking-space.github.io/AutoRLBench/contribute.html).

## What to send

Package your run as a directory named after the case. Nothing here has to be tidy — absolute
paths, hostnames and `broken`/`fixed` labels in your logs are expected; the packaging step
sanitizes them.

```text
<your_task>/
├── buggy_workspace/   code that reproduces the defect, plus that run's artifacts
├── fixed_workspace/   the same tree with the fix, plus its run's artifacts   (optional)
├── Dockerfile         the environment both runs were recorded on
└── README.md          the defect, the fix, and the measured outcome
```

**The two workspaces differ only by the fix.** Anything else that differs is something the
comparison is also measuring, and the threshold will be calibrated on a confounded pair.
Each workspace is self-contained: its own code, its own inputs, no sibling directory either
side reaches into.

**Each workspace carries its own run's artifacts**, where the run wrote them. At minimum:

- the full stdout/stderr of the run;
- the offline W&B directory, including the `run-*.wandb` binary (without it `wandb sync`
  cannot replay the run);
- every per-step metric file — they are small, and a threshold is read off them;
- a couple of rollout dumps as samples. Full rollout dumps are large and are not needed.

**The Dockerfile** pins versions and says in a comment which recorded run they came from.
Prefer one that builds without a build context.

**The README** states the defect, the fix as a diff, the measured outcome of both arms, and
a section for every plausible-but-wrong diagnosis. If you ran each arm once, say so: a
single-seed result supports margins, not variance, and whoever calibrates a threshold needs
to know which they have.

**Leave out** model weights, checkpoints, and anything derivable from what you did ship (a
FAISS index whose corpus is already in the bundle). Record how to obtain them instead.

Without a fixed run, the task is issue-only (diagnosis). With one, it also gets a solution
level.

### A worked example

[`tasks/environment/search_r1_zero_query_embedding`](tasks/environment/search_r1_zero_query_embedding/)
started life as raw material in this form.

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
└── README.md
```

What the README said, in brief:

- **The defect.** `Encoder.encode()` fills a host buffer, then writes the GPU embedding
  through `torch.as_tensor(buffer, device=cuda)`. Across devices that call copies instead of
  aliasing, so the write lands in a temporary and the function returns a zero vector.
- **Why it is silent.** Under inner-product search a zero query scores `0.0` against every
  document. FAISS returns a well-formed constant top-3, with no exception and no shape
  change. The run still trains and still improves — `val/searched` is `1.0` and `val/format`
  stays above 0.977 — only the retrieved text is unrelated to the question.
- **The measured outcome.** Same host, seed, data and 32-step budget; the two workspaces
  differ only in those four lines.

  | Step | buggy `val/em` | fixed `val/em` |
  | ---: | ---: | ---: |
  | 0 | 3.12% | 13.28% |
  | 20 | 11.72% | 31.25% |
  | **32** | **15.62%** | **32.81%** |

  Both arms improve, so "the run did not learn" is the wrong diagnosis. The gap is already
  open at step 0, which rules out every learning-side explanation.
- **Wrong diagnoses to reject.** The checkpoint, the reward, the data, the model size, or
  "training is not working".
- **The fix.** Replace the four lines with `query_emb.detach().cpu().numpy()` followed by
  `.astype(np.float32, order="C")`.
- **Non-fix differences.** The arms used different GPUs and retriever ports because they ran
  concurrently on one host. Those are placement parameters, not reproduction conditions.

The packaged result is the task linked above: one workspace, the fix sealed into the
solution oracle, and the run artifacts thinned and sanitized.

## Package it

Point an agent at the raw material with the package-task skill:

```text
Use the package-task skill to turn <your_task> into an AutoRLBench task under tasks/.
```

It produces `tasks/<domain>/<task_id>/` in the [task layout](README.md#task-layout).
Then check:

```bash
python3 -m autorlbench validate tasks/<domain>/<task_id>
```

It must report zero errors.

## Open a pull request

Open it against `main` with the generated task. Include the raw material only if it is small
enough to be useful as evidence.
