# Canonical issue oracle

Expected verdict: **unhealthy**.

The native compute_policy_loss subtracts detached current log probabilities instead of frozen behavior log probabilities; clipping becomes inactive during four optimizer passes over each rollout.

Fault location: `verl/trainer/ppo/core_algos.py`. Connect the faulty operation to actual observed evidence, distinguishing cause from symptoms. The dataset is pinned official GSM8K; benchmark injections are not claimed upstream bugs.

{
  "trainer_steps": 30,
  "initial_accuracy": 0.47,
  "final_accuracy": 0.0,
  "generated_training_completions": 3840,
  "mechanism_value": 0.15077247563749552,
  "passed": false,
  "rejection": "loss ignores the frozen rollout behavior policy"
}
