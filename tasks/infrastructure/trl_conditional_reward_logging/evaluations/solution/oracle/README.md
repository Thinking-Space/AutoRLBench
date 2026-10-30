# Reference repair

`src/trl/trainer/grpo_trainer.py`, `GRPOTrainer._generate_and_score_completions`: each rank iterates its own pending metric and extra-column keys before invoking collectives. A callback logs answer_parse_rate only on rank 0, so the ranks call different collectives at the same sequence number. Gather the union of keys first and iterate it identically on every rank. For metrics reduce the sum and count of actual supplied values, so an absent rank is not treated as a zero observation. The upstream repair also pads missing extra columns with None, but that unexercised path is not required for a passing diagnosis of this conditional-metric case.

Apply `repair.patch` from the workspace root. The accompanying results are from one control run.

The defective two-rank run raises a collective fingerprint mismatch before the first optimizer step. The control completes 24 optimizer steps and the final held-out evaluation, with reward 0.19140625. All 12 training answer_parse_rate values match the rank-0 rollout fractions, including 0.625 at the first generation event.

Both ranks must complete all 24 optimizer steps. The custom answer_parse_rate metric must remain present at all 12 generation events and equal the fraction of parsed answers on the rank that supplied it, within absolute and relative tolerance 0.000001. Disabling the metric or replacing an absent rank contribution with zero does not satisfy this criterion.
