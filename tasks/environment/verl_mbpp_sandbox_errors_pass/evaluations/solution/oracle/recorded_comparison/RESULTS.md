# Recorded comparison

`control/` is the defective workspace with `fix.diff` applied, run with the same seed, data and
60-step budget. Held-out pass rate is the grader's re-execution of each `val_generations` dump.

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective (`workspace/results/run60`) | 0.2786 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| control | 0.2966 | 0.3307 | 0.3587 | 0.3567 | 0.3527 | 0.3607 | 0.3667 |

Required score 0.33: margins 0.051 above the highest defective evaluation and
0.037 below the control at step 60.
