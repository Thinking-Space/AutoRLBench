# Solution verifier

The unchanged common `test.sh` invokes the standard schema-v2 `tests/grade_submission.py`. `score_mode=final` requires the logged `critic/score/mean` at step 200 rather than accepting the defective run's temporary earlier improvement. The captured step-120 defective score is 0.7109375, which would pass an inappropriate best-of-run 0.5 threshold.

The threshold is 0.5, calibrated against one defective final (-1.0) and one control final (0.6796875); margins are 1.5 and 0.1796875. Maximum 1.0 is the natural upper bound of this reward. No step-0 training score or multi-seed variance was recorded. The criterion measures recovery from the training collapse documented by the reward curve and rollout termination behavior.

A bare final value proves nothing, so the score is anchored four ways:

- **strict single-line extraction with run continuity** — a reading counts only when `step:<n> - ... critic/score/mean:<v>` appears on one line of one text log under `experiment_logs` (the console format verl's aggregate logger emits; nothing is concatenated across lines), and the supplying log must show at least `min_step_coverage` (0.95) of the training steps 1..200, so a fabricated or truncated log is not a run;
- **corroboration at step 200 from the same log** — `verl_tool/is_traj_finished/mean` ≥ 0.9 and `response_length/clip_ratio` ≤ 0.05; the recorded collapse ends at 0.0 / 0.5625, the control at 1.0 / 0.0;
- **required log patterns** — the staged configuration (training and held-out datasets, model, GRPO group size, the six tool-service launches, the historical action-stop-token wiring and log routing) must still be visible in the judged log, so swapping the data, model or budget fails even when the metric clears the bar;
- **live sha256 pinning** — with the stage environment (`AUTORLBENCH_WORKSPACE` / `AUTORLBENCH_VERIFIER`) the launch and logging chain is pinned against `grader_spec.json`, so editing the runner, the launcher or the console logger fails; the repair lives in the agent-loop reconstruction and is not pinned. A hand replay without the stage environment skips this check.

Only new files under `$AUTORLBENCH_RESULTS/experiment_logs` count. The verifier returns a binary score and the usual `experiment_metrics.json`, `metrics.json`, and `result.json`. The contributor must not submit copied or fabricated metric logs.
