# Recorded comparison

One run per side, seed 42, 100 steps, 1 x RTX PRO 6000, `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image (verlai/verl:vllm017.latest, same digest) through Apptainer. Held-out numbers re-score the dumped greedy responses (1,319 GSM8K test problems) with verl's strict rule (last `#### N` in the final 300 characters, compared as a string); training numbers come from the dumped rollouts (1,024 per step) and `metrics.jsonl`.

## Held-out (val_generations)

The solution grader's own rule is slightly more lenient (any last `#### <number>`, compared numerically, so `#### 18.00` counts). It scores step 100 at 0.493 (650 of 1,319) for the defective side and 0.566 (746 of 1,319) for the control; the threshold is 0.53.

| side | step | strict accuracy | logged val-core acc | mean characters | share with `####` | repetition loops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| defective | 0 | 0.001 | 0.001 | 971 | 0.023 | 1 |
| defective | 50 | 0.465 | 0.465 | 937 | 0.945 | 13 |
| defective | 100 | 0.486 | 0.486 | 928 | 0.980 | 5 |
| control | 0 | 0.001 | 0.001 | 967 | 0.022 | 2 |
| control | 50 | 0.569 | 0.569 | 890 | 0.973 | 3 |
| control | 100 | 0.566 | 0.566 | 891 | 0.973 | 0 |

Logged held-out accuracy at every evaluation:

| side | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective | 0.001 | 0.002 | 0.102 | 0.295 | 0.400 | 0.465 | 0.478 | 0.472 | 0.483 | 0.482 | 0.486 |
| control | 0.001 | 0.498 | 0.542 | 0.544 | 0.540 | 0.569 | 0.562 | 0.560 | 0.560 | 0.562 | 0.566 |

## Training signals (`metrics.jsonl`)

| side | step | `critic/score/mean` | `actor/entropy` | `response_length/clip_ratio` | `actor/kl_loss` | `actor/pg_clipfrac` | `actor/ppo_kl` | `actor/grad_norm` | `rollout_corr/kl` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective | 1 | 0.01172 | 0.5509 | 0.123 | 0.0004181 | 4.445e-05 | 5.183e-06 | 0.2176 | 0.0006193 |
| defective | 10 | 0.02637 | 0.5523 | 0.1191 | 0.001194 | 0.0001337 | 4.872e-05 | 0.3928 | 0.0005949 |
| defective | 17 | 0.08203 | 0.608 | 0.1533 | 0.006222 | 0.0003759 | -0.0002052 | 0.6237 | 0.0006357 |
| defective | 30 | 0.21 | 0.5433 | 0.08008 | 0.02095 | 0.0008536 | 7.222e-05 | 0.8495 | 0.0005744 |
| defective | 50 | 0.4443 | 0.3574 | 0.05664 | 0.04931 | 0.001068 | -1.911e-05 | 0.923 | 0.0005999 |
| defective | 100 | 0.5742 | 0.1555 | 0.01758 | 0.08918 | 0.00137 | -2.563e-05 | 0.8782 | 0.0004911 |
| control | 1 | 0.01562 | 0.5482 | 0.1074 | 0.001146 | 0.0003208 | 0.0006015 | 0.2819 | 0.0005491 |
| control | 10 | 0.4932 | 0.3389 | 0.0498 | 0.03867 | 0.001839 | 0.000276 | 0.8134 | 0.0004536 |
| control | 17 | 0.5742 | 0.2378 | 0.04395 | 0.05711 | 0.001754 | 0.0001865 | 0.7217 | 0.0004817 |
| control | 30 | 0.667 | 0.1424 | 0.02246 | 0.08012 | 0.002175 | 0.0001538 | 0.8112 | 0.0004541 |
| control | 50 | 0.624 | 0.1124 | 0.02246 | 0.1035 | 0.002299 | 0.000417 | 0.7918 | 0.0005318 |
| control | 100 | 0.6855 | 0.063 | 0.01953 | 0.119 | 0.002426 | 0.0004357 | 0.8847 | 0.0007923 |

## Training rollouts

| side | step | logged score | groups of 8 identical | zero-advantage groups | mixed-script garbage | mean characters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| defective | 1 | 0.012 | 0 | 116 | 4 | 1064 |
| defective | 50 | 0.444 | 0 | 40 | 4 | 860 |
| defective | 100 | 0.574 | 0 | 47 | 1 | 863 |
| control | 1 | 0.016 | 0 | 113 | 5 | 1053 |
| control | 50 | 0.624 | 0 | 62 | 4 | 855 |
| control | 100 | 0.686 | 0 | 70 | 10 | 838 |

"Mixed-script garbage" counts responses with at least five CJK, kana, Hangul, Cyrillic or Arabic characters; "repetition loops" counts responses in which a 2 to 40 character unit repeats 12 or more times in a row.
