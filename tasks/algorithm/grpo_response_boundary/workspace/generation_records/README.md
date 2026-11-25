# Generation records

`engine-requests.json` contains both requests of an archived pre-update
Nemotron 4B generation batch. It includes the submitted prompt token IDs,
returned token IDs, actual sampled log probabilities, and completion status.
The run used a 16,384-token response budget.

`heldout-responses.json` contains two response records for the same question
from an independent BF16 Nemotron 4B AIME 2024 evaluation. It contains the
original response text, generation counts, and scores. The evaluator recorded
the counts and did not retain the exact token arrays for these two responses.

These are archived generation records. The GRPO launcher below uses the
prepared AIME 2024 split and performs its own rollout.
