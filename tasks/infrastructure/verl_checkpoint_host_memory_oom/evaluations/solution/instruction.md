# Repair the training experiment

Inspect the source, configuration and frozen run evidence in `/workspace`. Determine why an initially running GRPO experiment fails, explain the causal chain in `/workspace/report.md`, and repair it.

The fixed workload uses Qwen2.5-0.5B-Instruct and the complete official GSM8K splits: 7,473 training questions and 1,319 held-out test questions. Train exactly 10 steps and save the checkpoint at step 10. Validation runs before training and after checkpoint restoration; there is no validation pass between the tenth update and saving. Keep the model, data, seeds, reward, batch sizes, rollout count, sequence lengths, learning rate, optimizer passes, evaluation schedule and checkpoint schedule unchanged. Keep the fixed 73,836 vLLM KV-cache blocks unchanged. All model, optimizer, scheduler, RNG and dataloader state must be saved. Do not skip checkpoint contents, use reduced precision for saved state, or extend training beyond step 10.

The resource budget is one A800 80 GB GPU and the fixed container RAM limit recorded in the environment and frozen cgroup evidence. Do not increase that limit, disable Ray's memory monitor, fabricate metrics, or modify the frozen evidence or verifier. You may repair workspace source and configuration, including state placement, while preserving public interfaces and unrelated behavior.

Do not launch training. Submit `/workspace/report.md` describing the diagnosis, supporting evidence, and repair. Do not use the network. The verifier starts fresh training, checks actual checkpoint files, then restores step 10 in a new process and runs held-out validation without additional training updates. Accuracy gain is not the acceptance criterion.
