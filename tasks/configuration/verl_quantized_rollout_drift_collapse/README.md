# The fp8-quantized rollout engine re-quantizes the trainer's bf16 weights on every sync, and the drift compounds: a GRPO run that is healthy for ~70 steps collapses to zero reward and stays there, with fluent rollouts the whole way down

Training Qwen/Qwen2.5-1.5B-Instruct with verl GRPO on Countdown with a throughput-tuned vLLM rollout engine (online fp8 weights, bf16 trainer): the run is healthy for ~70 steps, then `critic/score/mean` slides monotonically to 0.0 and never recovers, while the rollouts stay fluent, well formatted, and keep quoting the prompt's numbers the whole way down.

| | |
| --- | --- |
| **Author** | Xuyang Yuan |
| **Affiliation** | Shanghai Jiao Tong University |
| **Type** | Configuration |
| **Domain** | Math RL |
| **Tags** | `quantized-rollout` `fp8` `delayed-training-collapse` `off-policy-drift` |
| **Framework** | verl v0.8.0 (`7aed6b2`, shipped in the workspace), GRPO, vLLM 0.11.0 |
| **Run time** | ~4.2 hours (300 steps) |
| **Resources** | 2 × H100 (80 GB) |

## Difficulty

`run_grpo.sh`'s throughput block turns on `actor_rollout_ref.rollout.quantization=fp8`: the
vLLM rollout engine holds online fp8-blockwise-quantized weights while the trainer stays
bf16, and every weight sync re-quantizes the trainer's fresh bf16 tensors back into the
engine (`verl/utils/vllm/vllm_fp8_utils.py`, `load_quanted_weights`). This is a real verl
feature ("online FP8 quantization" for colocated rollouts) and an ordinary throughput
decision; nothing it does raises an error. (Upstream's fp8 sync path cannot run on this
stack at all — `environment/prepare.sh` applies one file-level fix to the vendored verl
so the quantized-rollout configuration is runnable; the fix is identical in every
recorded arm and is environment repair, not the defect.)

The failure is a delayed statistical collapse with no broken component, and every
observable is designed by accident to mislead:

- **A 50-step budget cannot see it.** Two 50-step seeds of this exact config look healthy
  (final `critic/score/mean` 0.584 / 0.514 vs the bf16 control's 0.631 / 0.549 — within a
  seed's spread). The collapse starts around step 70 and completes by step 78.
- **Nothing crashes, and the degradation is staged.** Through the healthy phase and the
  slide (dumped steps 1/59/72/78/84): 0 empty responses, 0 `!`-only degenerate responses,
  entropy ~0.6, no NaN; at step 84 — six steps past the collapse — 63% of responses still
  close an `<answer>` tag and still open with "To create an equation using the numbers 26,
  23, 39, and 20 that equals 97, …" quoting the prompt's own numbers, so the dump reads
  like a healthy run that simply stopped solving anything. Only much later, starved of any
  reward, does the never-trained policy drift into tag-repetition babble (steps 150/300:
  509/512 and 506/512 responses still close an `<answer>` tag, but the insides are
  mixed-script fragments like `(69-x-y-x+y#endif)`; still 0 empty, 0 `!`-only).
- **The likely diagnoses are all wrong.** "Model too weak / data too hard" and "RL reward
  eventually collapses / RLHF wall" both die on the control: the identical seed, data, and
  300-step budget stays healthy at every step (final 0.678, never below 0.44 after
  warmup). "RL reward
  eventually collapses / RLHF wall" dies the same way. "fp8 kills generation quality"
  dies on the rollout dumps — the fluency and format adherence are intact on both sides of
  the collapse; what shifted is the sampled policy's ability to solve, not its ability to
  write. And the corruption family from the sibling tasks (`fp8_kv_scale_sleep`'s `!!!`
  spam, `quantized_rollout_restart`'s whitespace) is absent here: this run never sleeps the
  engine (`free_cache_engine=False`) and never re-quantizes corrupted weights — each sync
  starts from the trainer's exact bf16 tensors.

The mechanism is off-policy drift: the sampler `q` (fp8 shadow) systematically differs
from the trained policy `p` (bf16), the mismatch is re-injected at every weight sync, and
the group-centered GRPO update cannot correct a bias that lives in the sampler rather than
in the advantage (the paper's TIM setting; see the oracle). The severity ladder (all arms
share seed, data, budget; only the precision knobs differ) brackets the effect:

| arm | outcome |
| --- | --- |
| bf16 rollout, seed 1 / seed 2, 50 steps | healthy, final 0.631 / 0.549 |
| fp8 weights, seed 1 / seed 2, 50 steps | healthy-looking, final 0.584 / 0.514 |
| **fp8 weights, seed 1, 300 steps (this run)** | **healthy to ~step 70 (peak 0.535), 0.0 from step 78 to 300** |
| bf16 rollout, seed 1, 300 steps (control) | healthy at every step, final 0.678 |
| fp8 KV cache arms (with or without fp8 weights), 50 steps | a different family: reward 0 from step 1, prompt-blind responses |

A rescore probe (`reference/p1_kl_mismatch.py`) shows the per-token engine-vs-trainer gap
is small in the mean (+0.0005 nats; bf16 sanity +0.0007) but noisy (per-token std 0.17) —
consistent with a slow-compounding drift rather than an instant break, which is why the
50-step runs pass and the 300-step run does not.

The intended horizon is the full 300 steps the recorded run completed; the run disables
in-run validation (`trainer.test_freq=-1`) and dumps every step's rollouts
(`trainer.rollout_data_dir`), so the per-step training score plus those dumps are the
per-step signal, and the independent evaluation is the control/ladder evidence in
`evaluations/issue/reference/` (a bf16 run of identical seed, data, and budget).

## Reference solution

- **issue**: name the quantized rollout configuration — `rollout.quantization=fp8` with a
  bf16 trainer, whose every weight sync re-quantizes bf16 → fp8 into the engine so the
  sampler keeps tracking a quantized shadow of the policy, and the accumulated off-policy
  drift collapses the run past the ~70-step mark. Attributions that must fail: the reward
  function, model capability or data difficulty, learning rate and schedule, "RL runs
  eventually collapse", and corruption-style explanations (`!`-spam, whitespace, engine
  state loss) that the fluent rollout dumps contradict.

Issue-only: several equally valid repairs (disable the weight quantization, a milder
quantization tier, score centering) restore training, and the collapse arm ran once.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.
