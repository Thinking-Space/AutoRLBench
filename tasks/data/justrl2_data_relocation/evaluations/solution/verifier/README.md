# Joint assessment and continuation grade

The hidden expected verdict describes the starting implementation. `report.md`
needs that verdict and a nonempty explanation. Solution grading does not judge
the prose semantically; the issue evaluation does.

Score 1 requires both correct assessment and the fresh continuation audit. The
same joint envelope is written to result.json and metrics.json. Reports must be
regular bounded files; output writes replace symlinks atomically. A failed
verifier cannot reuse an earlier success file.

Only separate Docker stages are supported. See the repository security review
for confirmed attacks and residual limits.
