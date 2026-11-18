# Recorded comparison

Controls are the unmodified recipe (the defective workspace with `fix.diff` applied), seeds 42 and
43: the control runs shipped with `tasks/infrastructure/verl_fsdp_bf16_master_weights`. The shipped
evidence is the seed-42 defective run.

Held-out accuracy, re-scored from `val_generations/<step>.jsonl` (last number of the response vs the
gold answer):

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 (shipped) | 0.4610 | 0.4678 | 0.4723 | 0.4541 | 0.4579 | 0.4549 | 0.4511 |
| defective, seed 43 | 0.4617 | 0.4776 | 0.4973 | 0.4996 | 0.4845 | 0.4867 | 0.5057 |
| control, seed 42 | 0.4503 | 0.4996 | 0.4973 | 0.5254 | 0.5413 | 0.5375 | 0.5580 |
| control, seed 43 | 0.4602 | 0.4905 | 0.5027 | 0.5353 | 0.5239 | 0.5557 | 0.5406 |

`actor/grad_norm` over 60 steps: defective 0.12-0.36 (seed 42) and 0.16-0.41 (seed 43); control
0.58-0.94 and 0.55-0.93.
