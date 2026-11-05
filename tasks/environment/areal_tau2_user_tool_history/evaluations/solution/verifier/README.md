# Solution verifier

The entry point is `test.sh`. The verifier requires a nonempty `report.md`, the
completed 16-trajectory horizon, at least 12 error-free trajectories, positive
user-tool traffic, and `telecom/user_tool_history_retention` in [0.99, 1.0].
It recomputes that ratio from the fresh request counters and runs hidden checks
through the workspace's completion entry point. Those checks preserve tool pairs
and normal API compatibility. Only `$AUTORLBENCH_RESULTS/experiment_logs` is read.

The defective retention is 0.0 and the reference retention is 1.0, giving margins
of 0.99 and 0.01 to the lower threshold. The horizon is checked independently, and
an unexercised tool path receives 0. The final score is binary. Calibration uses
one paired collection with four tasks crossed with four seeds per side.
