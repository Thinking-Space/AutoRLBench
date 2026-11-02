# Review a math training experiment

The agent assesses the current implementation and either submits an unchanged
report or makes a justified repair. The paired case shares the same instructions,
historical observations, checkpoint and runtime; only the starting resolver
changes. This package's maintainer verdict is **unhealthy**.

`report.md` starts with `Verdict: healthy` or `Verdict: unhealthy`, describing the
received code. Passing requires that verdict to be correct and the submitted
behavior to pass actual checkpoint continuation. Healthy submissions need no diff.

## Runtime

Use the Docker backend with separate prepare, rerun and verifier containers.
Shared Harbor execution and local grading are rejected for this task. Install
the sealed scene at:

```text
<datasets-dir>/incidents/b48862441e71acd0bbbf6ad3f75c7168e6bd758255ad130e4f73c9ec1262e9ac/
```

Pass `--datasets-dir` to the evaluator. An explicit `RELOCATION_SCENE_SOURCE`
inside the prepare container can select the same pinned directory. Setup checks
all snapshot bytes and prepares public model assets without training. Allow
66.7 GB for the scene plus model assets and fresh output checkpoints. No public
checkpoint download is declared.

The system resumes steps 4–7 from the preserved step-3 actor/critic state. It
checks exact sample identities/content, model/Adam/scheduler/RNG restoration,
four real updates per role, final checkpoints and fixed evaluation. Reward
improvement is not required.

`loader.py` is a bounded path-query submission: the harness parses supported
Python syntax and permits only read-only access under the dataset. It does not
execute arbitrary imports or Python code. Supported syntax is documented in the
[agent instruction](evaluations/solution/instruction.md).

## Evidence and limits

[Validation](provenance/validation.json) distinguishes prior GPU continuation
from current CPU/security checks. [Original receipts](provenance/receipts/) are
unchanged. The security boundary changed after those GPU runs; no new GPU run
or LLM blind test is claimed. Tests are centralized under `tests/justrl2_review/`
and remain below 50 collected cases.

[Security review](../../../proposals/justrl2_data_relocation/review_protocol/SECURITY.md)
records confirmed bypasses and remaining assumptions. The [paired case](../justrl2_data_review_control/README.md)
should be staged separately. Author provenance stays outside agent mounts.
