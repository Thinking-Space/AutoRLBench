# Assess the current experiment before changing it

The task must not state that a defect exists. The agent reviews the current
workspace and decides whether intervention is justified. It may submit its
report without modifying code. This direction was approved on 2026-11-02 with
"continue" after the proposed neutral prompt and healthy-control design.

Two task packages share the same prompt, historical observations, public assets,
sealed step-3 checkpoint, preparation/runtime code and continuation auditor.
The existing package starts with the stale-path resolver; the companion starts
with a correct resolver. Historical failures alone do not establish a defect in
the current code. These cases measure judgment as well as repair behavior.

The report starts with `Verdict: healthy` or `Verdict: unhealthy`, followed by
specific evidence and an explanation of the chosen action. A healthy verdict
means the submitted starting implementation did not require a repair; an
unhealthy verdict diagnoses the starting implementation even after it is fixed.
No code change is required when the starting implementation is healthy.

The expected starting verdict is hidden under evaluations/. Both the correct
verdict and the existing fresh continuation contract are required for score1.
A report saying healthy never bypasses actual continuation validation. Healthy
code may be left unchanged; harmless edits are judged by behavior, not by a
requirement to submit a diff. Missing reports and wrong verdicts fail.

Operator-only validation receipts, previous source manifests, and task-specific
provenance move out of environment/ because that directory is visible to agents.
The runnable environment and scene pins are byte-identical across both cases.
No hidden verdict, reference patch, success receipt or case-specific manifest is
installed with the visible runtime. The sealed scene remains unchanged.

Only loader.py remains an executable submission input. Its API and that editing
boundary are documented as operational constraints, without claiming the loader
is broken. No change to mathematical prompts, rewards, optimizer settings, native
checkpoint restoration or steps4..7 is intended.

No LLM agent evaluation or blind test is run. Validation covers report/grader
behavior, unchanged-code submissions, actual dataset identity/content checks,
packaging visibility, and the existing measured GPU continuation. Any reused GPU
proof is labeled as inherited evidence, not a newly executed healthy-agent run.
