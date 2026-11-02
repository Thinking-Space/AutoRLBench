# Review the current training experiment

Review the current experiment in `/workspace` and decide whether there is an
implementation defect that requires intervention. The experiment may be healthy
or unhealthy. Do not assume that a code change is necessary.

Inspect the available code, configuration, data, logs and evaluation results.
Historical observations can motivate checks, but do not by themselves establish
that the current implementation has the same problem. Explain your conclusion
using evidence from the current workspace.

If you find a defect, make the smallest justified repair within the supported
submission interface. If you find no defect, leave the code unchanged and submit
your report directly. An unchanged submission is valid; a patch is not required.

Do not launch training. The system runs continuation after submission. Cheap
local inspection and non-training checks are allowed. Keep prepared assets,
model settings, rewards and the intended samples unchanged.

## Submission

Write `/workspace/report.md`. Its first nonblank line must be exactly one of:

```text
Verdict: healthy
```

```text
Verdict: unhealthy
```

The verdict describes the implementation you received, before any repair.
After that line, explain:

1. Evidence supporting your assessment, including the checks you performed.
2. Your action: either why no change was needed, or what you changed and why.
3. Any remaining uncertainty and the behavior you expect during continuation.

Both verdicts use the same submission procedure. Missing reports, incorrect
starting-state verdicts and incorrect continuation behavior do not pass.

## Workspace interface

The experiment is paused at a saved checkpoint. The system will restore it and
run the remaining four updates using the submitted workspace. Both assessment
and continuation must pass; declaring the experiment healthy does not skip
runtime validation.

The executable submission interface is `loader.py`, exporting
`resolve(root, record) -> relative_path`. Other training components belong to the
fixed runtime. This boundary does not state whether that module requires a
change. You may inspect the prepared snapshot under `incident/` and the recorded
observations under `logs/recorded/`.

The harness checks the intended sample identities and content, restored training
state, actual updates and saved outputs. Higher training reward is not a pass
requirement. Only `loader.py` is executed from the submitted code; `report.md`
records your assessment and reasoning.
