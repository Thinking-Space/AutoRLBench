# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and frozen evaluation evidence. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Use held-out evaluation evidence together with rollout, retrieval, runtime,
optimization, and training signals. Training reward alone does not establish
held-out health.

## Your job

If you find a defect, fix it in the staged workspace at `/workspace`. **Do not
launch training.** Do not start the experiment, in the foreground or
background, and do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. Cheap, non-training checks are fine.

**Do not use the network.** Do not fetch code, documentation, packages, or any
other material from outside the staged workspace.

When you are done, submit a report at `/workspace/report.md` summarizing:

1. **Root cause** — the defect and the workspace evidence connecting it to the
   observed behavior.
2. **Changes** — each file you changed and what you changed in it.
3. **Expected effect** — the behavior your repair should change.

A missing, empty or symlinked `/workspace/report.md` scores 0.

## Supported submission interface

The executable repair interface consists of the body of
`catalog.py::build(rows, passages)` and the `training_catalog` field in
`config.json`. The catalog selection must remain `catalog_001` or `catalog_002`.
Other configuration values, the producer's imports and command-line wrapper,
training/runtime source, prepared assets, scorer and frozen evidence are fixed.
Do not edit the parquet, JSONL or index assets directly. The system builds a
fresh effective catalog and index from the submitted interface after submission.

`build` is a pure document transformation. It receives valid question rows and
source passages and returns a list of documents with string `id` and `contents`
fields. Its output must contain exactly the supplied source passages, preserving
their contents, identities and multiplicities, including on additional valid
inputs with different row counts. Document order may vary.
The system checks only the producer selected by the effective training path;
an unused producer is not executed merely to require an unrelated edit.

The producer body supports assignments, `if`, `for`, returns, comprehensions,
container literals, indexing, comparisons, arithmetic, f-strings and raises.
Available calls are `list`, `dict`, `set`, `tuple`, `enumerate`, `len`, `str`,
`range`, `min`, `max`, `zip`, `all`, `any`, `sorted`, `sum`, `isinstance`,
`ValueError`, `TypeError`, and the methods `get`, `strip`, `append`, `extend`,
`copy`, `items`, `keys`, `values`, `split`, `join`, `startswith`, `endswith`,
`replace` and `re.findall` (including the `re.S` flag). Imports, file access,
networking, reflection, decorators, recursion and unbounded loops are unsupported. Source size, AST,
operation, process-time and memory budgets are enforced.

## What the system will run

After submission, a task-specific verifier executes a fresh CPU check. There
is no training rerun and no LLM judge in this solution evaluation.

The verifier checks fixed inputs and the submitted interface, constructs the
effective retrieval catalog from the original prepared sources, builds its
index mapping using the prepared source embeddings, and launches the actual
production retrieval service in a fresh directory. Both training and validation
retrieval must preserve valid source passages and return working e5/FAISS
top-k results. Empty catalogs, missing sources, disabled retrieval and changed
scorers or runtime files do not pass.

The verifier computes its own results. Existing passing logs or JSON summaries
are not evidence of a repair; do not copy, edit or fabricate them. All required
functional checks must pass for score 1. This solution assesses data-pipeline
repair and retrieval functionality, without requiring an increase in training
reward or held-out accuracy.
