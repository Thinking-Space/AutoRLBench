# Implemented task: SQL-agent return propagation across tool observations

**Status:** full issue-diagnosis task submitted for maintainer review. The original proposal's solution tier remains conditional; no calibrated performance threshold is claimed.

The maintainer invited full construction after reviewing this proposal. The completed package is [`tasks/algorithm/verl_reinforcepp_observation_credit`](../tasks/algorithm/verl_reinforcepp_observation_credit/README.md), with a pinned runtime, reproducible preparation, historical code and evidence, the canonical issue verifier, a hidden oracle, and reviewer-only control artifacts.

The implementation records a real three-update Qwen3-4B REINFORCE++ run over bounded, read-only warehouse SQL. Four successful tool trajectories contain 2,480 earlier action tokens whose raw returns are all zero. Counterfactual replay of the same full batches through the upstream repair restores those returns. The paired GPU control also preserves earlier reward.

The generated workload includes one-to-many joins, refunds, cancellations, shipments, strict result checks, and hidden database perturbations. It has 512 training cases, 12 recorded development cases (96 supported), and 128 sealed cases. All 28 CPU workload checks pass.

Both arms use one matched seed and three optimizer updates on 2 × H100 80 GB. The short runs verify the mechanism but do not establish improved development accuracy. The earlier proposal's three-seed, 60-update campaign and a performance-based solution tier were not completed. Sealed policy evaluations were not run. Future extension requires a repaired endpoint above step zero and a defensible separation from the defective arm.

The benchmark validator and the actual Dockerfile/preparation checks pass. Three unrelated pre-existing image tests still fail, as documented in the [verification record](../tasks/algorithm/verl_reinforcepp_observation_credit/reference/verification.md). No judge-model evaluation has been run. The earlier design remains available in this PR's commit history; this document and the task README describe the delivered scope.
