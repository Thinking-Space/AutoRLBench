# Reference repair

Keep the sampler's log probabilities as the denominator of the clipped policy
loss. Extend the training scorer to credit a final boxed answer that exactly
matches a training label. The reference patch changes the launched actor
setting and adds the same boxed-answer fallback used in the 30-step control.

The shared AIME24 evaluator scored the initial weights at 31/240, the original
run at 41/240, and the repair at 59/240. In steps 21 through 30 the original run
logged 23 positive training rewards and zero median clipping. The reference
repair logged 158 positive rewards and a median clip fraction of about 0.159.
The denominator-only run logged 46 rewards, and the scorer-only run had zero
clipping. All results are from one training run per condition.
