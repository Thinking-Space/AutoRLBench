# Joint assessment and continuation grade

The hidden expected verdict describes the starting implementation. `report.md`
needs that verdict and a nonempty explanation. Solution grading does not judge
the prose semantically; the issue evaluation does.

Score 1 requires correct assessment, unavailable-record probes on both relocated
and existing paths, and the fresh continuation audit. The
same joint envelope is written to result.json and metrics.json. Reports must be
regular bounded files; output writes replace symlinks atomically. A failed
verifier cannot reuse an earlier success file.

Only separate Docker stages are supported. See the task's
[security review](../../../provenance/SECURITY.md) for confirmed attacks and
residual limits.
