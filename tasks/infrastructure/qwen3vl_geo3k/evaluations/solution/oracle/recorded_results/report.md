# Qwen3-VL Geo3K: full measured report, buggy vs fixed

8x H800; the defective arm and the reference arm each ran 20 steps, one experiment per arm.

Main results: the 20-step mean training–inference KL drops from **0.006590862 to 0.000623229 nat/token**, a reduction of about **10.58x**; final correct answers **355/601 → 370/601**, accuracy **59.07% → 61.56%**.
At step 0 the two arms answered 343 and 353 questions correctly; over 20 steps the net gains were 12 and 17, so the final 15-question gap cannot be attributed entirely to the training repair.
The buggy arm did not collapse either; its 20-step mean training reward is even slightly higher than fixed. The strongest discriminating evidence is training–inference consistency, not short-run reward.

## Setup and data

Qwen3-VL-8B-Instruct, Geo3K, GRPO, FSDP2; prompt 2048, response 8192, context 10240, dynamic token budget 16384.
Train batch 32 with 5 rollouts per prompt, i.e. 160 samples per step; each validation covers 601 questions, n=1.
Actor lr=1e-6, KL coefficient=0.01; both arms start from the original checkpoint, resume disabled.
Upstream commit `483b8a009ba3a97563edee3a19887e4862b8094a`. Both arms share the same environment, including Accelerate 1.13.0.

## Metric interpretation

Metric definitions and limitations:

- `rollout_corr/kl` = mean(log π_rollout − log π_actor) over valid response tokens; a training–inference KL estimated on rollout samples, in nat/token. It is not a maximum absolute difference.
- `rollout_corr/k3_kl` = mean(exp(Δ) − Δ − 1), where Δ = log π_actor − log π_rollout. This is another KL estimate and must not be conflated with the max error of the frozen tests.
- `rollout_corr/log_ppl_abs_diff` first takes mean(log π_rollout − log π_actor) per sequence, then the absolute value, averaged across sequences. It is not mean(abs(token log-prob difference)); positive and negative differences within a sequence can cancel.
- `training/rollout_probs_diff_*` uses absolute differences of probabilities after exp(log-prob), so it has no nat unit; the Pearson correlation is likewise computed on probabilities.
- `actor/kl_loss` is the actor's KL loss against the reference — a different comparison target from the training–inference KL; `actor/ppo_kl` is also not the training–inference KL.
- The per-step token mean/max absolute log-prob difference was not saved, nor are the raw per-token log-prob arrays, so they cannot be reconstructed from probability differences. The six-case frozen test did save token log-probs, whose mean/max can be computed.
- Geo3K reward = 0.9 x answer correct + 0.1 x format correct; in this run every validation and rollout file contains only scores of 0 and about 0.9, with no format bonus. The log `acc` reuses the reward score, not the usual binary accuracy. The correct-answer counts in the tables were verified against the scores of the 601 validation JSONL records.
- Validation runs only at steps 0, 5, 10, 15, and 20; other steps have no validation numbers, and no interpolation or backfilling from previous numbers was done.
- B/F are two independent trainings: same initial checkpoint and hyperparameters, yet the initial validation results are not identical; the final difference cannot be fully attributed to the repair, and same-step results must not be treated as per-token pairings of identical generation samples.
- All cross-step means are simple averages, not recomputations over all tokens of all steps pooled together.
- Gradients stayed finite and both arms completed, but this is a single 20-step short run; it does not demonstrate long-term RL collapse or generalization across seeds. Passing the evaluation thresholds does not mean all token log-probs are identical.


## Validation-set results per evaluation

| Step | buggy correct / 601 | fixed correct / 601 | buggy accuracy | fixed accuracy | buggy log acc score | fixed log acc score |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 343 | 353 | 57.07% | 58.74% | 0.513644 | 0.528619 |
| 5 | 368 | 363 | 61.23% | 60.40% | 0.551082 | 0.543594 |
| 10 | 350 | 356 | 58.24% | 59.23% | 0.524126 | 0.533111 |
| 15 | 353 | 367 | 58.74% | 61.06% | 0.528619 | 0.549584 |
| 20 | 355 | 370 | 59.07% | 61.56% | 0.531614 | 0.554077 |

Initial → final: buggy **+2.00 percentage points**; fixed **+2.83 percentage points**.
The buggy arm's best result is at step 5 (368/601, 61.23%); fixed peaks at step 20 (370/601, 61.56%).

## Training statistics overview

| Metric (simple 20-step mean) | buggy | fixed | B / F |
| --- | --- | --- | --- |
| Training–inference KL estimate (nat/token) | 0.006591 | 0.000623 | 10.575 |
| Training–inference K3 KL estimate (nat/token) | 0.006609 | 0.000629 | 10.510 |
| Absolute difference of sequence-mean log-prob (nat/token) | 0.010003 | 0.001240 | 8.067 |
| Token probability absolute difference | 0.011607 | 0.004915 | 2.362 |
| Token probability Pearson | 0.993060 | 0.999253 | 0.994 |
| Training reward | 0.548156 | 0.545062 | 1.006 |
| Gradient norm | 0.410747 | 0.462388 | 0.888 |
| KL loss against reference | 0.002305 | 0.001908 | 1.208 |
| entropy | 0.404657 | 0.407207 | 0.994 |
| PPO clip fraction | 0.000428 | 0.000365 | 1.173 |
| response tokens | 2476.377826 | 2425.425934 | 1.021 |
| response clip ratio | 0.074688 | 0.070312 | 1.062 |
| Training step seconds (excluding separately counted testing) | 118.486284 | 116.424794 | 1.018 |

## Training–inference consistency per step

B=buggy, F=fixed; slashes separate the two arms' metrics at the same step.

| Step | KL nat/token B / F | K3 KL nat/token B / F | Sequence log-ppl absolute difference B / F | Mean probability difference B / F | Probability Pearson B / F |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.005229 / 0.000602 | 0.005125 / 0.000605 | 0.008869 / 0.001145 | 0.010695 / 0.004880 | 0.994282 / 0.999199 |
| 2 | 0.008670 / 0.000692 | 0.008055 / 0.000663 | 0.011780 / 0.001294 | 0.013174 / 0.005074 | 0.991271 / 0.999175 |
| 3 | 0.007136 / 0.000608 | 0.006831 / 0.000628 | 0.009553 / 0.000990 | 0.012468 / 0.005045 | 0.993298 / 0.999287 |
| 4 | 0.005917 / 0.000551 | 0.005948 / 0.000629 | 0.008002 / 0.001098 | 0.011633 / 0.004977 | 0.993863 / 0.999282 |
| 5 | 0.006287 / 0.000669 | 0.006509 / 0.000638 | 0.012000 / 0.001539 | 0.011250 / 0.004844 | 0.992837 / 0.999208 |
| 6 | 0.006930 / 0.000682 | 0.006844 / 0.000657 | 0.009659 / 0.001218 | 0.012043 / 0.004961 | 0.992966 / 0.999187 |
| 7 | 0.006185 / 0.000544 | 0.006077 / 0.000641 | 0.011610 / 0.001207 | 0.011331 / 0.005098 | 0.993346 / 0.999246 |
| 8 | 0.006393 / 0.000542 | 0.006037 / 0.000588 | 0.008715 / 0.000992 | 0.011096 / 0.004683 | 0.993764 / 0.999285 |
| 9 | 0.007811 / 0.000515 | 0.007802 / 0.000612 | 0.009573 / 0.001287 | 0.012625 / 0.004895 | 0.991954 / 0.999348 |
| 10 | 0.005743 / 0.000602 | 0.005609 / 0.000606 | 0.010018 / 0.001671 | 0.010636 / 0.004754 | 0.993872 / 0.999261 |
| 11 | 0.007555 / 0.000573 | 0.007516 / 0.000650 | 0.009390 / 0.001376 | 0.013134 / 0.004981 | 0.991699 / 0.999229 |
| 12 | 0.005202 / 0.000620 | 0.005158 / 0.000620 | 0.007949 / 0.001192 | 0.010573 / 0.004873 | 0.994334 / 0.999252 |
| 13 | 0.006210 / 0.000628 | 0.006069 / 0.000594 | 0.008328 / 0.001094 | 0.011085 / 0.004707 | 0.993427 / 0.999296 |
| 14 | 0.006278 / 0.000587 | 0.009793 / 0.000656 | 0.008918 / 0.001134 | 0.011227 / 0.005024 | 0.993611 / 0.999287 |
| 15 | 0.006834 / 0.000632 | 0.006403 / 0.000619 | 0.010335 / 0.001303 | 0.011499 / 0.004826 | 0.992730 / 0.999267 |
| 16 | 0.007048 / 0.000663 | 0.007088 / 0.000636 | 0.013331 / 0.001191 | 0.011848 / 0.004969 | 0.992136 / 0.999261 |
| 17 | 0.007328 / 0.000699 | 0.006921 / 0.000639 | 0.009227 / 0.001459 | 0.012149 / 0.005011 | 0.992279 / 0.999264 |
| 18 | 0.006821 / 0.000649 | 0.006472 / 0.000610 | 0.011738 / 0.001157 | 0.011249 / 0.004806 | 0.992875 / 0.999265 |
| 19 | 0.006191 / 0.000696 | 0.005973 / 0.000651 | 0.009670 / 0.001194 | 0.011593 / 0.004921 | 0.993210 / 0.999245 |
| 20 | 0.006049 / 0.000712 | 0.005941 / 0.000636 | 0.011405 / 0.001258 | 0.010838 / 0.004963 | 0.993451 / 0.999223 |

## Training and gradients per step

| Step | Reward B / F | Grad norm B / F | Reference KL loss B / F | Entropy B / F | Response tokens B / F |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.3994 / 0.3938 | 0.3110 / 0.4210 | 0.000268 / 0.000271 | 0.4122 / 0.3988 | 3066.6 / 2930.7 |
| 2 | 0.4894 / 0.4894 | 0.5261 / 0.4020 | 0.000926 / 0.000703 | 0.4110 / 0.4073 | 2031.1 / 1965.2 |
| 3 | 0.4725 / 0.4556 | 0.4697 / 0.4297 | 0.000751 / 0.000665 | 0.4407 / 0.4342 | 2898.3 / 3099.4 |
| 4 | 0.4612 / 0.4387 | 0.4327 / 0.3370 | 0.000971 / 0.000794 | 0.4248 / 0.4232 | 2674.7 / 2627.5 |
| 5 | 0.6187 / 0.5287 | 0.4592 / 0.5396 | 0.001948 / 0.000956 | 0.3925 / 0.3860 | 1882.9 / 1898.5 |
| 6 | 0.4781 / 0.5513 | 0.4676 / 0.9129 | 0.001845 / 0.001046 | 0.4312 / 0.4187 | 2343.2 / 2201.9 |
| 7 | 0.5231 / 0.4950 | 0.3103 / 0.6307 | 0.002351 / 0.001104 | 0.4150 / 0.4238 | 3217.2 / 2891.9 |
| 8 | 0.4612 / 0.4837 | 0.4755 / 0.3187 | 0.002257 / 0.001164 | 0.4039 / 0.3908 | 3406.0 / 3143.1 |
| 9 | 0.6469 / 0.6525 | 0.4122 / 0.3009 | 0.003427 / 0.001624 | 0.4143 / 0.4110 | 2167.0 / 2270.0 |
| 10 | 0.6019 / 0.5906 | 0.3841 / 0.6825 | 0.003161 / 0.001840 | 0.3962 / 0.3939 | 2280.9 / 2184.2 |
| 11 | 0.5456 / 0.5737 | 0.5272 / 0.2874 | 0.002801 / 0.002260 | 0.4049 / 0.4096 | 1842.3 / 1641.3 |
| 12 | 0.5512 / 0.5063 | 0.3902 / 0.3382 | 0.002556 / 0.001965 | 0.4080 / 0.4008 | 2762.8 / 2413.9 |
| 13 | 0.4838 / 0.5062 | 0.3885 / 0.3664 | 0.002146 / 0.001920 | 0.3934 / 0.3957 | 2787.7 / 2844.0 |
| 14 | 0.6412 / 0.6356 | 0.2814 / 1.0770 | 0.002569 / 0.002514 | 0.4150 / 0.4171 | 2465.5 / 2351.5 |
| 15 | 0.6412 / 0.5794 | 0.2890 / 0.3693 | 0.002911 / 0.003255 | 0.3869 / 0.4010 | 2237.4 / 2295.6 |
| 16 | 0.7200 / 0.7088 | 0.3282 / 0.6280 | 0.003040 / 0.003085 | 0.3926 / 0.4134 | 2185.0 / 2208.8 |
| 17 | 0.6075 / 0.6244 | 0.3289 / 0.3291 | 0.002957 / 0.003637 | 0.3895 / 0.4028 | 2054.6 / 2228.5 |
| 18 | 0.4612 / 0.4612 | 0.3214 / 0.3259 | 0.003227 / 0.003066 | 0.3820 / 0.4047 | 2547.3 / 2403.2 |
| 19 | 0.5681 / 0.6131 | 0.3806 / 0.2624 | 0.002629 / 0.002984 | 0.3934 / 0.4080 | 2428.0 / 2508.9 |
| 20 | 0.5906 / 0.6131 | 0.7313 / 0.2889 | 0.003361 / 0.003310 | 0.3856 / 0.4031 | 2249.0 / 2400.3 |

## Truncation and timing per step

| Step | Response clip ratio B / F | PPO clip fraction B / F | Step seconds B / F | Validation seconds B / F |
| --- | --- | --- | --- | --- |
| 1 | 0.1125 / 0.1312 | 0.000357 / 0.000253 | 140.2 / 134.9 | — / — |
| 2 | 0.0688 / 0.0625 | 0.000594 / 0.000379 | 112.8 / 108.7 | — / — |
| 3 | 0.0688 / 0.0875 | 0.000514 / 0.000419 | 125.0 / 121.2 | — / — |
| 4 | 0.0750 / 0.0812 | 0.000436 / 0.000207 | 119.0 / 116.3 | — / — |
| 5 | 0.0500 / 0.0437 | 0.000686 / 0.000451 | 109.2 / 111.0 | 162.5 / 165.8 |
| 6 | 0.0688 / 0.0500 | 0.000303 / 0.000514 | 116.1 / 111.7 | — / — |
| 7 | 0.1500 / 0.0875 | 0.000421 / 0.000382 | 126.6 / 121.0 | — / — |
| 8 | 0.1562 / 0.1187 | 0.000351 / 0.000432 | 126.5 / 125.2 | — / — |
| 9 | 0.0500 / 0.0625 | 0.000482 / 0.000448 | 114.7 / 115.8 | — / — |
| 10 | 0.0750 / 0.0688 | 0.000425 / 0.000390 | 113.2 / 112.6 | 176.9 / 176.4 |
| 11 | 0.0562 / 0.0437 | 0.000476 / 0.000310 | 111.6 / 106.9 | — / — |
| 12 | 0.0750 / 0.0750 | 0.000329 / 0.000318 | 118.4 / 113.0 | — / — |
| 13 | 0.0812 / 0.0812 | 0.000264 / 0.000333 | 121.8 / 117.0 | — / — |
| 14 | 0.0312 / 0.0500 | 0.000472 / 0.000367 | 114.2 / 111.5 | — / — |
| 15 | 0.0688 / 0.0437 | 0.000494 / 0.000343 | 116.6 / 111.9 | 178.5 / 153.8 |
| 16 | 0.0688 / 0.0562 | 0.000367 / 0.000481 | 116.9 / 112.9 | — / — |
| 17 | 0.0562 / 0.0750 | 0.000467 / 0.000278 | 111.1 / 112.4 | — / — |
| 18 | 0.0812 / 0.0625 | 0.000283 / 0.000307 | 115.9 / 117.8 | — / — |
| 19 | 0.0500 / 0.0625 | 0.000489 / 0.000382 | 116.8 / 117.7 | — / — |
| 20 | 0.0500 / 0.0625 | 0.000357 / 0.000306 | 122.9 / 129.0 | 174.0 / 147.9 |

Total training-step time: buggy **39.50 minutes**, fixed **38.81 minutes**; the extra testing at steps 5/10/15/20 totals **11.53 / 10.73 minutes** respectively.
The above excludes initial validation, model loading, environment checks, and the frozen parity capture; the whole qualification, from log start to completion, took about **122 minutes**.
Mean response length is about 2.4K tokens, but roughly **7%** of responses still hit the 8192 cap; the cap has not eliminated truncation of long answers.
`response/aborted_ratio` is 0 throughout for both arms.
The actor-reported peak allocated GPU memory is buggy **59.73 GB** vs fixed **59.60 GB**, with reserved peaks **70.36 / 70.32 GB**; these are actor metrics and must not be read as the memory peak of the whole job.

## Six-case frozen test of the pre-training model

64 fixed response tokens per case; both arms are scored on exactly the same captured data.

| Case | buggy max / mean (nat) | fixed max / mean (nat) | Position deviation B / F |
| --- | --- | --- | --- |
| text_control | 0.228682 / 0.008360 | 0.228682 / 0.008360 | 0 / 0 |
| single_image_256 | 0.901937 / 0.066213 | 0.478671 / 0.026023 | 63 / 0 |
| one_turn_two_images_128 | 1.443999 / 0.106773 | 0.248286 / 0.014292 | 27 / 0 |
| two_turn_two_images_128 | 1.059987 / 0.099186 | 0.318377 / 0.023957 | 27 / 0 |
| one_turn_two_images_320x224 | 3.740287 / 0.148000 | 0.550462 / 0.026297 | 129 / 0 |
| single_image_heldout_512x384 | 0.888529 / 0.078771 | 0.236807 / 0.017878 | 191 / 0 |

These max=3.740/0.550 nat results were obtained before training and must not be treated as the parity of the step 20 checkpoint.
The fixed arm retains tail-end error; all cases meet the max absolute difference < 1.0 nat criterion and the position coordinates are fully consistent. This does not mean all token log-probs are identical.

## Evidence files

- `all_metrics.csv`: 42 rows (both arms, steps 0–20), 98 raw metric columns; missing values left blank.
- `per_step_summary.csv`: main training metrics per step.
- `metric_summary.csv`: each metric's mean, min, max, and last value plus observation count.
- `validation.csv`: raw score, correct-answer count, and binary accuracy per validation.
- `frozen_parity.csv`: errors and position comparison of the six-case frozen test.
- `comparison.png` / `comparison.svg`: trend charts of the measured data above.

Defective-arm log: `workspace/logs/recorded_run/train.log` (relative to the task root).
Reference-arm log: `evaluations/solution/oracle/control_run/train.log` (relative to the task root).
The logs retain the W&B/DataLoader messages from exit cleanup; both arms completed training and validation.
