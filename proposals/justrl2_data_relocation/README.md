# Review a math-RL data pipeline

A storage migration can silently remove difficult questions from training when
sample-ID paths become stale and failed reads are replaced with other samples.
In the recorded fault, read failures reached 50%, difficult examples disappeared
from admitted samples, and mean reward rose from 0.406 to 0.703.

The paired tasks ask the agent to assess the current implementation before
changing it. Historical failure logs do not determine the current verdict:

- [Faulty case](../../tasks/data/justrl2_data_relocation/README.md): stale resolver;
  diagnose and repair it.
- [Healthy control](../../tasks/data/justrl2_data_review_control/README.md): current
  resolver already works; a justified unchanged submission is valid.

Both use the same sealed step-3 actor/critic state. The fixed engine resumes
steps 4–7 and checks original sample IDs/content, actual restored optimizer state,
four updates and final checkpoints. Reward improvement is not a pass threshold.

The [protocol and security review](review_protocol/README.md) describe assessment,
execution isolation and remaining limits. [Measured native evidence](incident/evidence/README.md)
is separate from later CPU/security validation. No LLM blind test was run.

`evidence/recorded-audit.tar.gz` is the single retained archive of the original
three-arm reproduction. Verify and recompute it with Python 3.12:

```bash
python3 check_recorded.py --out /tmp/reproduction.json
```

`run_case.py`, `runtime/`, public asset preparation and `incident/build.py` remain
because they construct/reproduce the frozen checkpoint. The obsolete task grader,
duplicated unpacked logs, old process plans and repeated validation histories
have been removed. Current tests live only in `tests/justrl2_review/`.
