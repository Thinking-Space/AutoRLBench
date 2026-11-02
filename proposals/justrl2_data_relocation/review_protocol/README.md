# Review before repair

Two tasks share a checkpoint, dataset, historical observations and neutral
instructions. The starting `loader.py` differs: one uses a stale path, the other
already resolves current locations. The agent reports whether intervention is
needed. Healthy code may be submitted unchanged.

The first report line is `Verdict: healthy` or `Verdict: unhealthy`. Solution
score requires the correct starting verdict and a passing fresh continuation.
A nonempty explanation is required, but its semantic quality is assessed only
by the separate issue judge. No patch, reward increase or fixed difficulty ratio
is required.

The continuation checks exact logical IDs/content, complete native checkpoint
restoration, four actor/critic updates, final checkpoint readback and fixed
evaluation. Prior GPU receipts are retained as historical execution evidence.
They do not validate later security changes or a newly run healthy agent.

Use separate Docker stages. Shared Harbor execution and local grading are
rejected because they allow agent processes to affect the verifier. Author
provenance and expected verdicts are not mounted into the agent stage.

Submitted Python is parsed as a bounded path-query language, never imported or
executed. It supports `json.loads`, `Path`, local assignments, conditions, bounded
loops/list comprehensions, data indexing and read-only dataset path operations.
See the task instruction for supported syntax. [SECURITY.md](SECURITY.md) records
confirmed attacks, fixes and limitations.

Tests are maintained only in `tests/justrl2_review/`, with fewer than 50 collected
cases including parameter expansion. `dataset_validation.json` records exact
reads against the frozen dataset; no LLM blind test or agent success rate is
claimed.
