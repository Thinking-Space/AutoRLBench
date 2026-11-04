# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

`src/trl/trainer/grpo_trainer.py`, `GRPOTrainer._generate_and_score_completions`: each rank iterates its own pending metric and extra-column keys before invoking collectives. A callback logs answer_parse_rate only on rank 0, so the ranks call different collectives at the same sequence number. Gather the union of keys first and iterate it identically on every rank. For metrics reduce the sum and count of actual supplied values, so an absent rank is not treated as a zero observation. The upstream repair also pads missing extra columns with None, but that unexercised path is not required for a passing diagnosis of this conditional-metric case.

## Observable impact

Every item in this section is derivable from the shipped workspace
(`train.sh`, `run.py`, `results/`). The counterfactual control run is
author-side reference and lives in Documentation reconciliation below.

- The defective two-rank run raises a collective fingerprint mismatch before
  the first optimizer step: at SequenceNumber 18, rank 0 runs `_ALLGATHER_BASE`
  on a shape-[1] tensor while rank 1 runs `BROADCAST` on a shape-[64] tensor
  (`results/train.log`).
- train.sh requests 24 optimizer steps; the run freezes at 0/24 with zero
  optimizer updates — the traceback fires in `_prepare_inputs` at step 0,
  `results/` contains no `metrics_rank*.jsonl`, no checkpoints and no
  `final_metrics.json`, and `exit_code.txt` is 1.
- The only recorded generation event (step 0) left three rollout rows per rank
  in `rollouts_rank{0,1}.jsonl`: on rank 0 two of three completions parse to a
  final answer (exactly one of all six scores reward 1.0), on rank 1 none
  parse, and the four unparsed completions end without a committed final
  answer (mid-arithmetic or mid-expression).

## Trigger conditions

The recorded run has two GPU processes and a reward metric emitted only on rank 0.

## Documentation reconciliation

- Control reference (author-side; the same reproduction with the conditional
  logging repair from upstream PR #7382 applied — not shipped in the workspace,
  and comparisons to it are not evidence items): the control completes all 24
  optimizer steps and the final held-out evaluation, with reward 0.19140625;
  all 12 training `answer_parse_rate` values match the rank-0 rollout
  fractions, including 0.625 at the first generation event.

Reject diagnoses of GPU out-of-memory, inaccessible model weights, a generic network failure, bad arithmetic rewards, or insufficient model capacity. The collective mismatch is a communication-order consequence of process-local key sets. Disabling logging hides the defect and loses required diagnostics. This run directly exercises conditional metrics. The related extra-column fix is included from upstream, but no separate extra-column reproduction is claimed.

## Source

The defective workspace is TRL commit 03b22f5f93868a8aa73c5089ab99fbceb05ae144. The control applies the conditional logging repair from upstream PR #7382, commit cc5bb3f8c94b7f9cf5c24029d7812e78849ccace. All experiment settings are identical between arms.
https://github.com/huggingface/trl/pull/7382
