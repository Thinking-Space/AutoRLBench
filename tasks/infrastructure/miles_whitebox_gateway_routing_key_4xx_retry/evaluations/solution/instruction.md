# Repair the rollout gateway

The workspace holds a GRPO training launch whose recorded run made zero
progress for thirty minutes: every rollout request is rejected before it
reaches the engine. The workspace carries the full miles source tree, the
rollout-side components the launch drives (`rollout/`, `recipes/`,
`scripts/`), the gateway the rollout talks through (`gateway/`), and the
routing proxy in front of the engine (`routing_proxy/`); under `recorded_run/`
are the driver, gateway, and routing-proxy logs of that run.

## Your job

Diagnose the failure, then repair the workspace at `/workspace` so the same
launch trains. **Do not launch training.** Do not start the experiment, in
the foreground or background, and do not use `nohup`, `&`, `tmux`, `screen`,
or any detach mechanism. The system runs the experiment itself after you
finish; anything you launch is wasted and scores nothing. Cheap,
non-training checks are fine (you may run the small components locally).

**Do not use the network.** Diagnose and repair using only what is staged
locally.

## The contract your repair must keep

The system re-runs `bash scripts/run_experiment.sh` on your workspace (a
20-minute window, 10 rollout iterations at 8 prompts x 4 samples) and grades
only that fresh run's `experiment_logs/summary.json`, plus sealed checks
described below. A passing repair:

- **the routing key survives the gateway hop.** The rollout sends
  `X-SMG-Routing-Key` on every request; after your repair, the routing proxy
  must accept them (zero `400 missing_routing_key` rejections in the fresh
  run's proxy log).
- **rollout iterations complete.** The fresh run must complete at least 5 of
  its 10 rollout iterations within the window.
- **deterministic 4xx fail fast.** The shared retry helper
  (`miles/utils/http_utils.py::_post`) must raise immediately on a 4xx other
  than 408/429, while still retrying transport errors, 5xx, 408, and 429.
  The verifier checks this directly against your workspace, independently of
  the training run.

These files are compared against pristine copies and must not be modified:
`routing_proxy/strict_router.py`, `rollout/gateway_generate.py`,
`recipes/run_training.py`, `scripts/run_experiment.sh`,
`data/dapo_math_17k_dedup.jsonl`, `recorded_run/`. Modifying them fails
grading outright. Everything else in the workspace is yours to repair.

When you are done, submit a report at `/workspace/report.md`: the verdict and
the evidence that decides it, each root cause you found with the evidence
connecting it to the recorded behavior, every file you changed and what you
changed in it. A missing or empty report scores 0.
