# Nucleus-sampled rollouts trained as if on-policy

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K. The launcher samples training rollouts with `top_p=0.5` "so a 0.5B model does not sample stray tokens", but the PPO update recomputes probabilities over the full softmax and applies no correction, so the half of the distribution that is never sampled is never pushed down. Entropy rises from 0.26 to 5.1, about a fifth of the rollouts derail into mixed-script garbage, and held-out accuracy peaks at step 10 (0.500) and ends at 0.486, against 0.566 for the same run with `top_p=1.0`.

| | |
| --- | --- |
| **Author** | Dengzhe Hou |
| **Domain** | configuration |
| **Tags** | `verl` `grpo` `gsm8k` `sampling` `top-p` `off-policy` `entropy` |
| **Model** | Qwen2.5-0.5B-Instruct |
| **Dataset** | GSM8K (openai/gsm8k main): 7,473 train / 1,319 test |
| **Framework** | verl 0.9.0.dev @ `6f2aa24`, FSDP + vLLM colocated, `run_gsm8k_grpo.sh` |
| **Environment** | `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` · 1 × GPU (recorded on 1 × RTX PRO 6000 96 GB) |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective + 1 control run, 100 steps, eval every 10 incl. step 0, seed 42. About 1 h each |

## Difficulty

The defect is one line in `run_gsm8k_grpo.sh`:

```bash
ROLLOUT_TOP_P=0.5            # nucleus sampling; at temperature 1.0 a 0.5B model otherwise samples stray tokens
...
    actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P
```

vLLM samples every training response from the top-50% nucleus of the policy at temperature 1.0.
The trainer then recomputes `old_log_probs` with the FSDP actor over the full softmax, and
`algorithm.rollout_correction` is off (`rollout_is: null`, `bypass_mode: false`), so the vLLM
log-probs feed only the `rollout_corr/*` metrics. GRPO treats the samples as if they came from the
full policy. Tokens outside the nucleus are never sampled, so they never receive a negative
advantage, while every negative-advantage update lowers the logits of sampled tokens and moves that
mass into the unsampled tail. The full-distribution entropy grows while the nucleus itself gets
narrower. Once the top token of a position falls below 0.5, the inflated tail enters the nucleus and
the response turns into random vocabulary until the 512-token cap.

Nothing fails and the first ten steps look like a healthy run: the policy learns the `#### N` format
and held-out accuracy reaches 0.500 at step 10 (control 0.498). The break comes at steps 13 to 17:
`actor/entropy` 0.63 → 2.25, `response_length/clip_ratio` 0.04 → 0.31, `actor/grad_norm` up to 3.7.
After that entropy never returns below 1.5 (peak 5.06 at step 63, 2.86 at step 100; the control
falls to 0.06), the reference KL loss climbs to 0.28 to 0.52 from step 26 on (control 0.12 at step
100), and 227 and 188 of 1,024 training
responses at steps 50 and 100 contain mixed-script garbage (0 at step 1, 4 to 10 in the control),
all scored 0. Sampling diversity collapses at the same time: 32 and 45 of 128 GRPO groups at steps 50
and 100 are eight byte-identical responses (0 in the control) and 109 to 111 groups have zero
advantage (62 to 70 in the control).

| step | 0 | 10 | 20 | 30 | 50 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| defective held-out strict accuracy | 0.001 | 0.500 | 0.475 | 0.446 | 0.463 | 0.486 |
| control held-out strict accuracy | 0.001 | 0.498 | 0.542 | 0.544 | 0.569 | 0.566 |
| defective `actor/entropy` | n/a | 0.366 | 1.635 | 3.221 | 3.250 | 2.860 |
| control `actor/entropy` | n/a | 0.339 | 0.195 | 0.142 | 0.112 | 0.063 |

The trap is that every downstream symptom has a familiar wrong explanation. Wrong diagnoses to
reject: the KL coefficient is too weak (0.001 in both runs; the control's KL stays at 0.12); the
512-token cap is too tight (the control's clip ratio is 2%; the cap hits come from garbage
continuations); the learning rate or entropy bonus (2e-6 and 0 in both runs); the reward or the data
(unchanged verl strict GSM8K rule and verl preprocessing, the same as the control); "the run is healthy,
held-out accuracy rose from 0 to 0.49" (that gain is the answer format, learned by step 10; the run
then falls below the control and its greedy held-out answers contain 175 repetition loops at step
100, against none in the control). The `rollout_corr/kl` of 0.086 at step 1, before any update
(control 0.0005), shows the sampler and the trained policy differ from the start.

## Reference solution

- **issue**: name `top_p=0.5` in the training rollouts, used by an update that treats the samples
  as on-policy (full-softmax `old_log_probs`, no rollout correction), as the root cause: the
  unsampled tail is never penalized, grows, and enters the nucleus as garbage. "Entropy explodes" or
  "the policy drifts" without the sampling mismatch earns partial credit. KL, length-cap, learning
  rate and healthy explanations fail.
- **solution**: sample training rollouts from the full distribution, `ROLLOUT_TOP_P=1.0`
  (`fix.diff`). Keeping `top_p<1` would need a correction for the truncated sampler; that variant is
  not recorded.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. A run passes only if the verdict score
  is 1.0 and the root-cause and evidence scores are both above 0.8.
- **solution**: the agent submits `report.md` and does not train. The harness reruns the experiment
  per `run_spec.json` (`run_gsm8k_grpo.sh`, 100 steps, 3 h budget) and grades only that run. The
  grader re-scores `experiment_logs/val_generations/100.jsonl` with a fixed strict rule (last
  `#### N` equals the gold answer, at least 1,300 rows) and requires ≥ 0.53. Defective 0.486 at step
  100 (best 0.500 at step 10), control 0.566 at step 100 and 0.001 at step 0, so fixing the
  launcher without the rerun fails. Margins 0.044 to the defective final and 0.036 to the control,
  calibrated on one run per side, so they are margins, not a variance estimate. The repair is in the
  runner, so the runner is not byte-compared (`runner_pinned: false`, as in
  `configuration/verl_grpo_ppo_mini_batch_too_small`); the GSM8K parquets and raw files are pinned
  by sha256 and `metrics.jsonl` must show training steps 0 to 100. Grader replay on the recorded
  evidence: defective exit 1, control exit 0. The recorded runs used the `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image (same digest) through
  Apptainer.

Not verified: a Docker build and run of the task (the runs and checks used Apptainer on the same image
digest), and a separate rerun of `fix.diff` (the control run is the workspace with `fix.diff` applied,
recorded once).
