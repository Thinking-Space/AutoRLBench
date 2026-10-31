# Recorded comparison

Two pairs of runs, one run per arm per pair, same host type (1 x RTX PRO 6000 Blackwell Max-Q),
same seed (`data.seed=2026`), data, model and 100-step budget. Within a pair the two workspaces
differ only in `calc_agent/calc_agent_loop.py` (`../fix.diff`).

- **Pair 1 (replicate_gsm8k_only/)**: held-out = 512 GSM8K test problems. Defective run
  12:28-14:30, control 15:25-19:17 on 2026-10-24. A first control attempt died at step 10 when
  its raylet was killed (host RAM at 111 GB of a 120 GB cgroup, or another job's `ray stop` on
  the node). It was discarded and rerun with more host memory. No other run was discarded.
- **Pair 2 (control/ + the task workspace)**: same recipe with GSM-Hard added as a second
  held-out set, because pair 1 showed no accuracy gap and the defective policy had stopped calling
  the calculator. GSM-Hard (GSM8K questions with large numbers) is where losing the calculator
  should cost accuracy. Both arms 18:17 onwards on the same node. Defective ended 20:36, control
  23:04.

Neither pair shows a reliable held-out accuracy gap. The mismatch metrics separate the arms at
every step.

| run | step | GSM8K acc | GSM-Hard acc | rollout_corr/kl | rollout/actor prob corr | num_turns/mean | response_length/mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective | 0 | 88.7% | 65.4% | 0.6667 | 0.029 | 3.90 | 177 |
| defective | 10 | 89.5% | 67.2% | 1.3513 | -0.016 | 3.84 | 93 |
| defective | 20 | 86.1% | 67.8% | 1.4349 | -0.015 | 3.42 | 102 |
| defective | 30 | 88.1% | 70.3% | 1.1980 | 0.017 | 3.08 | 129 |
| defective | 40 | 87.3% | 68.2% | 1.0372 | 0.026 | 2.83 | 169 |
| defective | 50 | 86.5% | 66.0% | 1.2688 | -0.009 | 2.61 | 125 |
| defective | 60 | 88.3% | 68.8% | 1.3025 | -0.010 | 2.48 | 130 |
| defective | 70 | 88.3% | 67.6% | 1.5919 | -0.000 | 2.55 | 132 |
| defective | 80 | 87.9% | 66.8% | 1.6864 | -0.006 | 2.36 | 136 |
| defective | 90 | 88.5% | 69.9% | 2.0210 | 0.005 | 2.20 | 115 |
| defective | 100 | 89.1% | 67.8% | 2.9860 | 0.007 | 2.07 | 94 |
| control | 0 | 88.3% | 65.6% | 0.0013 | 0.994 | 3.95 | 938 |
| control | 10 | 88.7% | 67.4% | 0.0014 | 0.994 | 3.95 | 875 |
| control | 20 | 89.1% | 68.0% | 0.0013 | 0.995 | 3.80 | 915 |
| control | 30 | 89.1% | 66.6% | 0.0014 | 0.994 | 3.61 | 830 |
| control | 40 | 88.9% | 67.0% | 0.0013 | 0.993 | 3.54 | 836 |
| control | 50 | 88.9% | 67.2% | 0.0013 | 0.992 | 3.62 | 786 |
| control | 60 | 89.5% | 66.0% | 0.0015 | 0.992 | 3.57 | 788 |
| control | 70 | 88.1% | 66.8% | 0.0015 | 0.990 | 3.66 | 799 |
| control | 80 | 89.8% | 65.8% | 0.0016 | 0.991 | 3.75 | 859 |
| control | 90 | 90.0% | 66.4% | 0.0016 | 0.990 | 3.80 | 788 |
| control | 100 | 88.5% | 67.8% | 0.0016 | 0.989 | 3.89 | 738 |
| defective (replicate) | 0 | 85.9% | n/a | 0.7918 | 0.025 | 3.96 | 160 |
| defective (replicate) | 10 | 88.3% | n/a | 1.3212 | -0.024 | 3.84 | 98 |
| defective (replicate) | 20 | 87.5% | n/a | 1.1777 | -0.003 | 3.11 | 120 |
| defective (replicate) | 30 | 87.3% | n/a | 1.3457 | -0.018 | 2.87 | 117 |
| defective (replicate) | 40 | 87.3% | n/a | 1.0219 | 0.027 | 2.53 | 160 |
| defective (replicate) | 50 | 87.7% | n/a | 1.6272 | -0.002 | 2.42 | 115 |
| defective (replicate) | 60 | 86.9% | n/a | 1.9768 | -0.008 | 2.27 | 119 |
| defective (replicate) | 70 | 88.5% | n/a | 2.2180 | -0.001 | 2.13 | 112 |
| defective (replicate) | 80 | 88.1% | n/a | 2.7463 | -0.012 | 2.04 | 125 |
| defective (replicate) | 90 | 86.3% | n/a | 3.6611 | -0.006 | 2.03 | 118 |
| defective (replicate) | 100 | 86.5% | n/a | 5.3869 | -0.009 | 2.00 | 101 |
| control (replicate) | 0 | 89.3% | n/a | 0.0014 | 0.994 | 3.92 | 942 |
| control (replicate) | 10 | 88.5% | n/a | 0.0013 | 0.993 | 3.89 | 878 |
| control (replicate) | 20 | 89.5% | n/a | 0.0015 | 0.994 | 3.52 | 864 |
| control (replicate) | 30 | 89.5% | n/a | 0.0016 | 0.993 | 3.06 | 723 |
| control (replicate) | 40 | 89.3% | n/a | 0.0014 | 0.993 | 3.03 | 775 |
| control (replicate) | 50 | 87.9% | n/a | 0.0016 | 0.992 | 2.88 | 722 |
| control (replicate) | 60 | 88.9% | n/a | 0.0016 | 0.991 | 3.01 | 667 |
| control (replicate) | 70 | 88.7% | n/a | 0.0016 | 0.989 | 3.04 | 680 |
| control (replicate) | 80 | 89.5% | n/a | 0.0017 | 0.990 | 2.98 | 727 |
| control (replicate) | 90 | 90.0% | n/a | 0.0015 | 0.989 | 3.24 | 703 |
| control (replicate) | 100 | 89.1% | n/a | 0.0016 | 0.988 | 3.29 | 713 |

Step-0 rows: accuracy is the pre-training eval. Mismatch columns show step 1 (the first update,
base weights). Tool calls per sampled training episode (episode dumps, every 97th prompt):
pair 2 defective 0.92 (steps 0-9) -> 0.00 (steps 90-99), control 0.92 -> 0.85. Pair 1
defective 0.96 -> 0.00, control 1.00 -> 0.55.

`drift_analysis/` holds the offline token/log-prob measurement on the 8 dumped episodes of the
first training step of each run (base weights, HF bf16 recompute), and the scripts that produced
it. Pair 2: defective keeps 4,417 of 16,684 sampled tokens (per-episode mean 85% dropped), KL on
surviving identical tokens 0.22 nats/token. Control round-trips all 17,583 tokens, KL 0.0014.
`rerender_with_reasoning.py` shows that keeping the reasoning in the transcript would reproduce
all but 2 of 17,583 control tokens (6 of 8 episodes byte-identical): the dropped reasoning is the
dominant drift here.
