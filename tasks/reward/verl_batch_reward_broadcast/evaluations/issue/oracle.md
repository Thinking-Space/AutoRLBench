# Canonical diagnosis

Expected verdict: unhealthy training pipeline.

case/reward.py reward_callback returns a singleton batch mean. collect_rewards broadcasts it to all responses. The GRPO group rewards become equal, advantages and policy gradients zero. Return individual scores and validate callback length. The DAPO adapter only replaces answer verification and retains this callback contract.

Root cause must be tied to source/configuration and recorded evidence; curve shape alone is insufficient.
