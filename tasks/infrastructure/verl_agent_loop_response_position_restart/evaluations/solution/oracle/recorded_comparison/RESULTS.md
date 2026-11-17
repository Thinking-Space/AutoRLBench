# Recorded comparison

`control/` and `control_seed1/` are the unmodified recipe (the defective workspace with `fix.diff`
applied), seeds 42 and 43, recorded once each in the shared
`verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image; they are the same control runs as
`tasks/infrastructure/verl_fsdp_bf16_master_weights`. `defective_seed1/` is a second defective run
(seed 43).

Held-out accuracy, re-scored from `val_generations/<step>.jsonl` with the grader's rule:

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 | 0.4632 | 0.4602 | 0.4564 | 0.4670 | 0.4875 | 0.4678 | 0.4776 |
| defective, seed 43 | 0.4511 | 0.4723 | 0.4716 | 0.4685 | 0.4852 | 0.4890 | 0.4670 |
| control, seed 42 | 0.4503 | 0.4996 | 0.4973 | 0.5254 | 0.5413 | 0.5375 | 0.5580 |
| control, seed 43 | 0.4602 | 0.4905 | 0.5027 | 0.5353 | 0.5239 | 0.5557 | 0.5406 |
