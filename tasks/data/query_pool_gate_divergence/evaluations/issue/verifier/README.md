# Issue verifier

The issue tier reads `$AUTORLBENCH_RESULTS/agent_answer.md` and compares it with the hidden
`oracle.md` using verdict / root-cause / evidence weights 0.2 / 0.6 / 0.2. The pass mark is 0.8.

The oracle carries six essential root causes, in two directions: the gate refuses pools the
project's own builder produces, *and* it starts runs whose configuration and data do not match
the pool it validated. A diagnosis that names only the visible rejections, or only the empty
run-tag component, cannot reach the mark. A flat "the gate is too strict", a green unit-test run,
or the gate's own `Validated mixed query pool` log line is not a root cause without the code path
that produces it. The judge uses six root-cause groups and four evidence groups;
visible source mechanisms, case-level replays, and small reproductions are
equivalent. Exact archived totals and duplicate counts are optional.
