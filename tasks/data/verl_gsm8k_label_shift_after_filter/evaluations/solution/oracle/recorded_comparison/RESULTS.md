# Recorded comparison

`control/` and `control_seed1/` are the unmodified recipe (the defective workspace with `fix.diff`
applied), seeds 42 and 43, recorded once each (seed 42 on an A100 40 GB, seed 43 on an A100 80 GB) in the shared
`verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image. The same control runs are the
reference for the other GSM8K tasks built from this recipe.

Held-out accuracy, re-scored from `val_generations/<step>.jsonl` with the grader's rule:

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective | 0.4488 | 0.4655 | 0.4534 | 0.4579 | 0.4632 | 0.4519 | 0.4443 |
| control, seed 42 | 0.4503 | 0.4996 | 0.4973 | 0.5254 | 0.5413 | 0.5375 | 0.5580 |
| control, seed 43 | 0.4602 | 0.4905 | 0.5027 | 0.5353 | 0.5239 | 0.5557 | 0.5406 |
