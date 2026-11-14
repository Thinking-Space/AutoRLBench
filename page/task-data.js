/*
 * Public task index. Each id is the task path under tasks/ in the repository.
 * Header metadata is copied from each task.toml by
 * scripts/generate_task_data.py; `brief` and `issue` are hand-written and
 * preserved across regenerations.
 */
window.AUTORLBENCH_TASKS = [
  {
    "id": "configuration/trl_gsm8k_eos_mismatch",
    "title": "TRL GSM8K GRPO with mismatched EOS configuration",
    "family": "Configuration",
    "tags": [
      "trl",
      "grpo",
      "eos",
      "gsm8k",
      "issue-only"
    ],
    "model": "Qwen3-1.7B-Base (GSM8K SFT)",
    "dataset": "openai/gsm8k",
    "framework": "trl",
    "image": "verl",
    "created": "2026-10-24",
    "license": "Apache-2.0",
    "brief": "Wrong EOS token id lets rollouts run past the answer.",
    "issue": "Training a GSM8K-SFT Qwen3-1.7B-Base with TRL GRPO. eos_token_id lists only endoftext (151643) and not <|im_end|> (151645), so rollouts continue past the answer and evaluation drops from 0.787 to 0.486."
  },
  {
    "id": "reward/slime_retool_length_reward_bonus",
    "title": "Length bonus reward hack",
    "family": "Reward",
    "tags": [
      "slime",
      "grpo",
      "reward-hacking",
      "length-bonus",
      "tool-call-parsing",
      "aime"
    ],
    "model": "Qwen3-4B",
    "dataset": "DAPO-Math-17k (RL) + ReTool-SFT (SFT)",
    "framework": "slime",
    "image": "slime-v0.3.2",
    "created": "2026-08-19",
    "license": "Apache-2.0",
    "brief": "A length bonus in the reward and a tool-call key mismatch make GRPO reward longer answers instead of correct tool use.",
    "issue": "Training Qwen3-4B with GRPO on DAPO-Math-17k (ReTool SFT warm-up). Two defects sit in examples/retool/generate_with_retool.py: reward_func adds an unconditional length bonus (5 * min(response_length / rollout_max_response_len, 1.0)) on top of the exact-match score, and the tool-call parser reads the \"parameters\" key while the chat template instructs {\"name\": ..., \"arguments\": ...}, so template-conforming code-interpreter calls carry no code and never execute. Logged eval/aime rises from +0.356 to +3.014, outside the healthy shaped-reward band [-1, 1], while eval response length grows from ~3.9k to 12.4k tokens, truncation from ~0.8% to 47.3%, and the decontaminated held-out score stays at about -0.77, bounding exact-match accuracy at ~11.6%."
  },
  {
    "id": "data/verl_taco_difficulty_mismatch",
    "title": "Training data is too hard",
    "family": "Data",
    "tags": [
      "verl",
      "grpo",
      "taco",
      "data-difficulty",
      "issue-only"
    ],
    "model": "Qwen2.5-1.5B-Instruct",
    "dataset": "TACO",
    "framework": "verl",
    "image": "verl:vllm017.latest",
    "created": "2026-10-21",
    "license": "MIT",
    "brief": "Training set is very hard while validation set is easy, so GRPO gets almost no learning signal.",
    "issue": "Training Qwen2.5-1.5B-Instruct with GRPO on TACO. The training problems are too hard for the model under the all-tests-pass reward, so training never learns and held-out accuracy stays flat."
  },
  {
    "id": "environment/search_r1_zero_query_embedding",
    "title": "Zero query vector in retrieval",
    "family": "Environment",
    "tags": [
      "verl",
      "grpo",
      "hotpotqa",
      "query_vector"
    ],
    "model": "Qwen2.5-1.5B-Instruct",
    "dataset": "HotpotQA distractor (1024 train / 128 held-out)",
    "framework": "verl",
    "image": "search-r1-verl",
    "created": "2026-10-16",
    "license": "Apache-2.0",
    "brief": "Query vector stays all zeros, retrieves same documents.",
    "issue": "Training Qwen2.5-1.5B-Instruct with GRPO on HotpotQA. The query vector stays all zeros, so every query retrieves the same documents."
  },
  {
    "id": "algorithm/verl_grpo_uid_after_repeat",
    "title": "GRPO uid assigned after rollout repeat",
    "family": "Algorithm",
    "tags": [
      "verl",
      "grpo",
      "advantage",
      "uid",
      "issue-only"
    ],
    "model": "Llama-3.2-1B-Instruct",
    "dataset": "DAPO-Math-17k",
    "framework": "verl",
    "image": "verl:vllm017.latest",
    "created": "2026-10-17",
    "license": "Apache-2.0",
    "brief": "GRPO silently degraded to baseline-free REINFORCE.",
    "issue": "Training Llama-3.2-1B-Instruct with GRPO on DAPO-Math-17k. The uid assignment was moved after the rollout repeat step, making every GRPO group a singleton and silently degrading training to baseline-free REINFORCE."
  },
  {
    "id": "infrastructure/verl_moe_token_normalization",
    "title": "MoE auxiliary gradients miss token normalization",
    "family": "Infrastructure",
    "tags": [
      "verl",
      "grpo",
      "moe",
      "gradient-normalization"
    ],
    "model": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "dataset": "openai/gsm8k main (7473 train rows, 64 held-out validation rows)",
    "framework": "verl",
    "image": "verl-moe-token",
    "created": "2026-10-18",
    "license": "Apache-2.0",
    "brief": "MoE z-loss gradients skip token normalization.",
    "issue": "Training Qwen1.5-MoE-A2.7B-Chat with GRPO on GSM8K. The per-token loss callback omits the token count, so MoE z-loss gradients are never divided back."
  },
  {
    "id": "algorithm/verl_grpo_advantage_broadcast",
    "title": "GRPO advantage broadcast along the sequence axis",
    "family": "Algorithm",
    "tags": [
      "verl",
      "grpo",
      "advantage",
      "taco"
    ],
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset": "TACO EASY (8368 train / 184 test)",
    "framework": "verl",
    "image": "verl",
    "created": "2026-09-24",
    "license": "MIT",
    "brief": "Per-sequence advantages broadcast over tokens, flattening training.",
    "issue": "Training Qwen2.5-1.5B-Instruct with GRPO on TACO EASY. compute_grpo_outcome_advantage broadcasts the per-sequence advantage along the sequence axis, so held-out accuracy stays flat around noise level for the whole run."
  },
  {
    "id": "infrastructure/qwen3vl_geo3k",
    "title": "Flat image positions from a raw AutoProcessor",
    "family": "Infrastructure",
    "tags": [
      "verl",
      "grpo",
      "qwen3-vl",
      "geometry3k"
    ],
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "dataset": "geometry3k (2101 train / 601 held-out)",
    "framework": "verl",
    "image": "verl-qwen3vl-cu130",
    "created": "2026-10-22",
    "license": "Apache-2.0",
    "brief": "Image position ids are computed outside the processor, skewing training.",
    "issue": "Training Qwen3-VL-8B-Instruct with GRPO on geometry3k. A raw AutoProcessor is used instead of the model's preprocessor, so image position ids come out flat and the KL against the reference model is an order of magnitude too high."
  },
  {
    "id": "infrastructure/torch_functional",
    "title": "Missing torch.roll in rm-pad log-prob computation",
    "family": "Infrastructure",
    "tags": [
      "verl",
      "grpo",
      "log-prob",
      "rm-pad"
    ],
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset": "TACO EASY (8368 train / 184 test)",
    "framework": "verl",
    "image": "verl",
    "created": "2026-09-09",
    "license": "MIT",
    "brief": "Log-probs are computed one token off after rm-pad.",
    "issue": "Training Qwen2.5-1.5B-Instruct with GRPO on TACO EASY. input_ids_rmpad_rolled is assigned as input_ids_rmpad instead of torch.roll(..., shifts=-1, dims=1), so old log-probs are shifted one token and dominate the surrogate loss."
  },
  {
    "id": "infrastructure/verl_qwen3_vl_grpo_training_case_v8k2",
    "title": "Qwen3-VL GRPO training case v8k2",
    "family": "Infrastructure",
    "tags": [
      "verl",
      "grpo",
      "qwen3-vl",
      "sequence-parallel",
      "issue-only"
    ],
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "dataset": "derek-thomas/ScienceQA",
    "framework": "verl",
    "image": "vl-grpo-cu129-recorded",
    "created": "2026-10-24",
    "license": "Apache-2.0",
    "brief": "Fused FSDP path drops the globally shifted SP labels.",
    "issue": "Training Qwen3-VL-8B-Instruct with GRPO on ScienceQA. The fused FSDP path does not consume the globally shifted sequence-parallel labels, so vLLM applies the wrong loss mask; correcting it is worth 5.35 percentage points of validation accuracy."
  },
  {
    "id": "reward/slime_search_r1_nomask",
    "title": "Search-R1 observations trained without a loss mask",
    "family": "Reward",
    "tags": [
      "slime",
      "grpo",
      "search-r1",
      "loss-mask",
      "retrieval-augmented"
    ],
    "model": "Qwen2.5-3B",
    "dataset": "Search-R1 nq_hotpotqa_train",
    "framework": "slime",
    "image": "slime-v0.3.2",
    "created": "2026-08-19",
    "license": "Apache-2.0",
    "brief": "Retrieved text gets gradient and observations render titles only, so the model stops searching.",
    "issue": "Training Qwen2.5-3B with GRPO on Search-R1 NQ/HotpotQA. Two defects sit in the rollout path: observation tokens are trained (loss_mask=1 on retrieved <information> text), and _passages2string drops retrieved passage bodies, so observations render titles only. Train response length collapses from ~1038 tokens to ~10 as the policy stops searching, and frozen eval/nq_test reaches only 16.7% at the 200-step horizon versus a 40.8% healthy-protocol baseline."
  },
  {
    "id": "reward/idea_awm_reward_diagnosis",
    "title": "Inaccurate rule-based task rewards in IDEA AWM OPSD",
    "family": "Reward",
    "tags": [
      "idea",
      "verl",
      "awm",
      "opsd"
    ],
    "model": "locaszzz/IDEA_sft",
    "dataset": "locaszzz/IDEA_training_data; locaszzz/AgentWorldModel-1K",
    "framework": "idea",
    "image": "idea-issue",
    "created": "2026-10-22",
    "license": "LicenseRef-IDEA4AutoRLBench-Composite",
    "brief": "The rule-based verifier awards full reward to trajectories that never complete the task.",
    "issue": "Training IDEA_sft with OPSD on the AgentWorldModel suite. The generated completion verifiers use incomplete acceptance predicates, so trajectories that do not satisfy the requested task still score full reward while held-out test score drifts down."
  },
  {
    "id": "infrastructure/verl_checkpoint_host_memory_oom",
    "title": "Checkpoint saving exhausts host memory at step 10",
    "family": "Infrastructure",
    "tags": [
      "verl",
      "grpo",
      "fsdp",
      "checkpoint",
      "optimizer-offload",
      "host-memory"
    ],
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "dataset": "openai/gsm8k main (7473 train / 1319 test rows)",
    "framework": "verl",
    "image": "verl_checkpoint_host_memory_oom",
    "created": "2026-10-27",
    "license": "Apache-2.0",
    "brief": "CPU-offloaded optimizer state leaves no headroom for the step-10 checkpoint save.",
    "issue": "Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K. CPU-offloaded Adam state leaves insufficient host-memory headroom for native checkpoint serialization at step 10, so container memory hits the fixed 20 GiB limit and the run dies."
  },
  {
    "id": "infrastructure/verl_lora_merge_base_weight_sync",
    "title": "Merged LoRA never reaches the rollout engine",
    "family": "Infrastructure",
    "tags": [
      "verl",
      "grpo",
      "lora",
      "fsdp2",
      "weight-sync"
    ],
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset": "openai/gsm8k main (7473 train rows, 1319 test rows)",
    "framework": "verl",
    "image": "verl",
    "created": "2026-10-25",
    "license": "Apache-2.0",
    "brief": "Every weight sync ships the base weights, not the merged LoRA.",
    "issue": "Training Qwen2.5-1.5B-Instruct with GRPO and a merged LoRA adapter on GSM8K. The merged tensors are gathered only after merged_lora_context has restored the base weights, so vLLM loads the base weights at every sync and held-out accuracy stays at the base level while the actor keeps training."
  },
  {
    "id": "reward/verl_math_verify_thread_zero_reward",
    "title": "Math-Verify scores zero inside reward workers",
    "family": "Reward",
    "tags": [
      "verl",
      "grpo",
      "math-verify",
      "reward-loop",
      "threading",
      "issue-only"
    ],
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset": "openai/gsm8k main + DigitalLearningGmbH/MATH-lighteval (7473 + 7500 train rows); validation GSM8K test (1319) + MATH-500 (497 of 500)",
    "framework": "verl",
    "image": "verl",
    "created": "2026-10-25",
    "license": "Apache-2.0",
    "brief": "The math verifier dies silently in reward worker threads, scoring every MATH answer 0.0.",
    "issue": "Training Qwen2.5-1.5B-Instruct with GRPO on a GSM8K + MATH mixture. math-verify's signal-based timeout raises ValueError in the reward loop's worker thread, and the bare except turns that into a 0.0 reward for every MATH sample, so MATH-500 logs exactly 0.0 at all 21 validations while GSM8K accuracy rises from 0.081 to 0.766."
  },
  {
    "id": "infrastructure/slime_critic_value_head_random_init",
    "title": "PPO critic value head randomly initialised instead of zeroed",
    "family": "Infrastructure",
    "tags": [
      "slime",
      "ppo",
      "critic",
      "value-head",
      "initialization"
    ],
    "model": "Qwen3-1.7B",
    "dataset": "Endless Terminals (1489 of 2492 tasks admitted; train 1039 / val 150 / test 300)",
    "framework": "slime",
    "image": "slime-v0.3.2",
    "created": "2026-10-14",
    "license": "Apache-2.0",
    "brief": "The critic's value head starts from random weights, inflating value loss ~77x.",
    "issue": "Training Qwen3-1.7B with Multi-Turn PPO on Endless Terminals. The critic's value head is drawn from a normal distribution instead of zeroed, so it opens with arbitrary value predictions and the critic value loss is inflated roughly 77x across the warmup window, most visibly after a reload where it reads like checkpoint corruption."
  },
  {
    "id": "infrastructure/miles_async_rollout_inflight_not_evicted",
    "title": "Fully-async GRPO in-flight rollouts kept alive across weight sync",
    "family": "Infrastructure",
    "tags": [
      "miles",
      "grpo",
      "async-rollout",
      "weight-sync",
      "throughput",
      "issue-only"
    ],
    "model": "Qwen3-1.7B",
    "dataset": "DAPO-Math-17k (dedup, 17917 prompts)",
    "framework": "miles",
    "image": "miles-async-rollout",
    "created": "2026-10-10",
    "license": "Apache-2.0",
    "brief": "In-flight rollouts decode to full length against stale weights, then never train.",
    "issue": "Training Qwen3-1.7B with fully-async GRPO on DAPO-Math-17k. The distributed weight-update mixin silently rewrites the configured abort pause mode to in_place, so in-flight rollout requests are kept alive across every weight sync, decode to full length against stale weights, and are then dropped — the engine burns decode on trajectories that never train, and the fixed 8-hour wall clock completes 92 training steps instead of 148."
  },
  {
    "id": "algorithm/trl_dapo_accumulation_normalization",
    "title": "GRPO generation and accumulation windows",
    "family": "Algorithm",
    "tags": [
      "trl",
      "grpo",
      "gsm8k"
    ],
    "model": "Qwen2.5-0.5B-Instruct",
    "dataset": "GSM8K, 256 training questions and 64 held-out questions",
    "framework": "trl",
    "image": "trl-grpo-torch26",
    "created": "2026-10-30",
    "license": "Apache-2.0",
    "brief": "The DAPO loss denominator skips the accumulation window, halving every update.",
    "issue": "Training Qwen2.5-0.5B-Instruct with GRPO and the DAPO loss on GSM8K. GRPOTrainer._compute_loss computes the DAPO denominator from the whole generation batch's token count without scaling it to the current gradient accumulation window (steps_per_generation=4, gradient_accumulation_steps=2), so every microbatch's loss and raw gradient are half the intended objective; the defective loss/reference ratio is about 0.5 against the control's 1.0, and the independent objective comparison passes 12 of 24 optimizer steps instead of 24 of 24."
  },
  {
    "id": "data/justrl2_data_relocation",
    "title": "Missing difficult examples inflate training reward",
    "family": "Data",
    "tags": [
      "justrl2",
      "miles",
      "ppo",
      "data-relocation"
    ],
    "model": "JustRL-II-base-model",
    "dataset": "UltraData-RL-2609 / Math (frozen public selection)",
    "framework": "miles / JustRL-II",
    "image": "justrl2-data-relocation",
    "created": "2026-10-31",
    "license": "Apache-2.0",
    "brief": "A moved hard-example directory leaves the training manifest stale, and reward climbs on the survivors.",
    "issue": "Training JustRL-II-base-model with PPO on a Math selection of UltraData-RL-2609. At the configured midpoint a storage migration moves the hard-example backing directory from live/hard to archive/hard without updating the manifest, so every hard sample is attempted, fails at the file-read boundary, and is dropped as ABORTED while the rollout loop refills the batch with easy examples — training, checkpointing and evaluation all stay alive, and observed reward rises because the surviving population is easier."
  },
  {
    "id": "configuration/verl_grpo_ppo_mini_batch_too_small",
    "title": "GRPO mini-batch far below the rollout batch: 32 updates per step",
    "family": "Configuration",
    "tags": [
      "verl",
      "grpo",
      "gsm8k",
      "ppo_mini_batch_size",
      "off-policy",
      "hyperparameter"
    ],
    "model": "Qwen2.5-0.5B-Instruct",
    "dataset": "GSM8K train 7,473 / held-out GSM8K test 1,319 (\\boxed{} prompt, Math-Verify reward)",
    "framework": "verl / GRPO",
    "image": "verl-0.9-vllm0.28",
    "created": "2026-10-29",
    "license": "Apache-2.0",
    "brief": "A 'smaller mini-batches to fit memory' setting multiplies optimizer updates per rollout batch eight-fold, and every later update trains on data from a policy that has already moved.",
    "issue": "Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K. run_gsm8k_grpo.sh sets PPO_MINI_BATCH_SIZE=4 ('smaller mini-batches to fit memory') against TRAIN_BATCH_SIZE=128 and ROLLOUT_N=8; verl multiplies ppo_mini_batch_size by rollout.n, so each 1,024-sample rollout batch is consumed in 32 optimizer updates of 32 samples instead of the recipe's 4 updates of 256. The later updates of every step are computed on a policy that has already moved: actor/ppo_kl stays 0.003-0.004 instead of settling near zero, entropy rises 0.35 -> 0.73 instead of decaying, responses lengthen and truncate (clip ratio 0.07 -> 0.37), the training score peaks at 0.62 (step 30) and falls back to 0.48 (control 0.75), and held-out accuracy peaks at 0.494 (step 20) and ends at 0.361 under the grader's boxed rule (0.387 by Math-Verify), below its own step 0; with ppo_mini_batch_size=32 the same workspace reaches 0.565, with 8 it reaches 0.541. No crash, no NaN, RUN_EXIT 0."
  },
  {
    "id": "algorithm/verl_pissa_residual_base_reference",
    "title": "PiSSA residual base used as the KL reference",
    "family": "Algorithm",
    "tags": [
      "verl",
      "grpo",
      "pissa",
      "lora",
      "kl-reference"
    ],
    "model": "Qwen/Qwen2.5-0.5B-Instruct + PiSSA rank 8",
    "dataset": "openai/gsm8k deterministic calibration subset (16 train, 8 validation)",
    "framework": "verl / GRPO",
    "image": "verl-vllm011",
    "created": "2026-11-01",
    "license": "Apache-2.0",
    "brief": "PiSSA initialization rewrites each adapted layer to W_residual = W_pretrained - A0*B0, so disabling the adapter exposes the residual weights — and the KL reference is computed against the wrong base.",
    "issue": "Training Qwen2.5-0.5B-Instruct with GRPO and a rank-8 PiSSA adapter. For a PiSSA residual-base checkpoint, disabling the adapter exposes W_residual rather than the original pretrained weights; the defective base-reference path therefore computes KL against the residual checkpoint. Final base-reference consistency is 0.04578 where the repaired path reaches 1.0, and the natural symptom is visible in the metrics: actor/kl_loss runs 0.76-0.98 across all four updates where a correct run logs near zero at step 1 (actor/base_model_kl around -4.6)."
  },
  {
    "id": "infrastructure/verl_pissa_vllm_adapter_request",
    "title": "PiSSA adapter omitted from vLLM rollout requests",
    "family": "Infrastructure",
    "tags": [
      "verl",
      "grpo",
      "pissa",
      "vllm",
      "policy-sync"
    ],
    "model": "Qwen/Qwen2.5-0.5B-Instruct + PiSSA rank 8",
    "dataset": "openai/gsm8k deterministic calibration subset (16 train, 8 validation)",
    "framework": "verl / GRPO",
    "image": "verl-vllm011",
    "created": "2026-11-01",
    "license": "Apache-2.0",
    "brief": "vLLM's list_loras() returns integer adapter IDs, but the generate path tests membership with the string name — the loaded PiSSA adapter is never attached, so every rollout samples the residual base.",
    "issue": "Training Qwen2.5-0.5B-Instruct with GRPO and a rank-8 PiSSA adapter on a fixed GSM8K subset. The vLLM request path gates the LoRA with lora_loaded = VLLM_LORA_NAME in loaded_loras, but engine.list_loras() returns the registered adapters' integer IDs, so the membership test \"123\" in {123, ...} is always false and every generation request omits the loaded adapter. vLLM samples the PiSSA residual base while the actor computes log probabilities with the adapter active: every rollout is off-policy, rewards and gradients are zero, and the run's final health check collapses (actor/rollout policy agreement 0.10476 versus 0.89072 after the one-line integer-ID fix)."
  },
  {
    "id": "algorithm/verl_policy_loss_padded_unmasked",
    "title": "Padded custom GRPO objective: unmasked loss",
    "family": "Algorithm",
    "tags": [
      "verl",
      "grpo",
      "recorded-run",
      "container-validation-pending"
    ],
    "model": "Qwen/Qwen3-1.7B",
    "dataset": "DAPO-Math-17k original 1,791,700 rows (no dedup); AIME24/25 30 questions each",
    "framework": "verl v0.6.1 / GRPO",
    "image": "verl-v061-grpo",
    "created": "2026-11-02",
    "license": "Apache-2.0",
    "brief": "A custom GRPO objective multiplies every token position — padded or not — by the full response length and averages the whole padded tensor, so with unpadding disabled the loss is dominated by almost pure padding.",
    "issue": "Training Qwen3-1.7B with GRPO on DAPO-Math-17k (150 steps, batch 512, mini 32, rollout 8). The registered luspo_token loss computes loss = (per_token_loss * lengths).mean() over the whole padded tensor without applying response_mask, replacing standard token-mean normalization and native dual clipping; recorded_config.yaml also sets use_remove_padding: false where the shared repaired reference runs true. actor/padding_fraction climbs 0.82 to 0.9955, actor/entropy collapses 210 to 0.80, response_length/mean falls from about 1500 to about 37 tokens, actor/grad_norm reaches 3.0e7, training reward peaks at 0.304 (step 38) and falls back to 0.090; AIME24/25 mean@32 at step 150 is 0.0031 versus 0.144 for the repaired reference."
  },
  {
    "id": "algorithm/verl_policy_loss_padded_mask_only",
    "title": "Padded custom GRPO objective: mask-only repair",
    "family": "Algorithm",
    "tags": [
      "verl",
      "grpo",
      "recorded-run",
      "container-validation-pending"
    ],
    "model": "Qwen/Qwen3-1.7B",
    "dataset": "DAPO-Math-17k original 1,791,700 rows (no dedup); AIME24/25 30 questions each",
    "framework": "verl v0.6.1 / GRPO",
    "image": "verl-v061-grpo",
    "created": "2026-11-02",
    "license": "Apache-2.0",
    "brief": "A custom GRPO objective sums masked token losses per response and averages over responses — a coherent, correctly masked objective, but not the standard token-mean target this benchmark declares.",
    "issue": "Training Qwen3-1.7B with GRPO on DAPO-Math-17k (150 steps, batch 512, mini 32, rollout 8). The registered luspo_token loss computes loss = (per_token_loss * mask).sum(-1).mean() — sequence-sum-then-average instead of token-mean aggregation — and omits native dual clipping; recorded_config.yaml also sets use_remove_padding: false where the shared repaired reference runs true. Training reward still rises (0.095 at step 1, peak 0.328 at step 46, 0.247 at step 150) but actor/grad_norm reaches 1.5e8 and AIME24/25 mean@32 at step 150 is 0.057 versus 0.144 for the repaired reference."
  },
  {
    "id": "infrastructure/verl_rollout_weight_overwrite",
    "title": "Initial checkpoint overwrites synchronized rollout weights",
    "family": "Infrastructure",
    "tags": [
      "verl",
      "grpo",
      "recorded-run",
      "container-validation-pending"
    ],
    "model": "Qwen/Qwen3-1.7B",
    "dataset": "DAPO-Math-17k original 1,791,700 rows (no dedup); AIME24/25 30 questions each",
    "framework": "verl v0.6.1 / GRPO",
    "image": "verl-v061-grpo",
    "created": "2026-11-02",
    "license": "Apache-2.0",
    "brief": "After each actor-to-vLLM weight sync, a post-sync hook reloads the initial checkpoint into the rollout engine — generation never uses the updated policy.",
    "issue": "Training Qwen3-1.7B with GRPO on DAPO-Math-17k (150 steps, batch 512, mini 32, rollout 8). case/sync.py finalize_weights reloads the initial safetensors checkpoint into the vLLM engine after Actor weights have been synchronized, so rollout generation repeatedly uses the initial weights while the Actor trains against its own updates. sync_trace.jsonl shows after_sync_max_abs_delta up to 4.9e-4 with changed_parameters rising to 154-156 of 226 sampled; actor/grad_norm grows 0.24 to 83.7 and actor/ppo_kl to 1.38 while critic/score/mean stays flat around 0.08-0.12; val-core/aime24/acc/mean@32 is exactly equal at step 0 and step 150 (0.0229) — held-out accuracy frozen at the step-0 level."
  },
  {
    "id": "algorithm/verl_reinforcepp_observation_credit",
    "title": "Tool observations erase earlier SQL action returns",
    "family": "Algorithm",
    "tags": [
      "verl",
      "tool-use",
      "issue-only"
    ],
    "model": "Qwen/Qwen3-4B",
    "dataset": "Generated operations SQL: 512 train, 12 pilot development, 128 sealed cases",
    "framework": "verl / REINFORCE++",
    "image": "tool-rl-vllm024",
    "created": "2026-11-03",
    "license": "Apache-2.0",
    "brief": "Qwen3-4B learns read-only warehouse SQL with REINFORCE++, but masked tool observations reset the return accumulator and remove terminal reward from earlier actions.",
    "issue": "Training Qwen3-4B with REINFORCE++ on generated read-only warehouse SQL (512 train rows, seed 17, 3 optimizer updates). In the defective return estimator, an internal zero in response_mask — a tool observation, not an episode boundary — clears the running return: of 192 recorded training trajectories, the four successful tool trajectories contain 2,480 earlier action tokens, every one receiving raw return zero despite terminal reward one, with whitened advantages about −0.0920 or −0.1041 (zero raw return does not mean zero advantage or zero gradient). Exact-batch replay through the repaired function restores those returns to one; an independently sampled paired control carries reward one across 587 earlier action tokens. Both arms have zero development accuracy at steps 0–3, so the evidence establishes the credit-assignment defect, not a measured accuracy gain."
  },
  {
    "id": "infrastructure/trl_tool_history_cross_contamination",
    "title": "Python repair continuations use another rollout's history",
    "family": "Infrastructure",
    "tags": [
      "trl",
      "tool-use",
      "issue-only"
    ],
    "model": "Qwen/Qwen3-4B",
    "dataset": "BugsInPy: 16 train, 4 development, 8 sealed bugs; two projects",
    "framework": "trl / GRPO",
    "image": "tool-rl-vllm024",
    "created": "2026-11-03",
    "license": "Apache-2.0",
    "brief": "Qwen3-4B learns real Python repairs with GRPO, but post-tool generation reuses the initial multiplicity and subsamples distinct histories, associating actions with the wrong rollout.",
    "issue": "Training Qwen3-4B with GRPO on BugsInPy Python repairs (16 train, 4 development, 8 sealed bugs across two projects; seed 17, 3 optimizer updates). After independently executed tools, generation takes every fourth input history and draws four outputs from each retained history: the defective run has 338/844 continuation mismatches across training and development evaluation versus 0/832 after the upstream fix, with 0/160 initial-input mismatches in each arm and 337/615 versus 0/357 among rows in naturally divergent continuation groups. Shapes remain valid and the model still repairs bugs (development success 4/16 trajectories at steps 0-3 in both arms), so correspondence rather than reward establishes the failure: replaying captured natural histories through both production dispatch functions gives 1/4 versus 0/4 mismatches, with identical-history and initial-generation controls both 0/4."
  },
  {
    "id": "configuration/verl_grpo_rollout_top_p_off_policy",
    "title": "Nucleus-sampled rollouts trained as if on-policy",
    "family": "Configuration",
    "tags": [
      "verl",
      "grpo",
      "gsm8k",
      "sampling",
      "top-p",
      "off-policy",
      "entropy"
    ],
    "model": "Qwen2.5-0.5B-Instruct",
    "dataset": "GSM8K (openai/gsm8k main): 7,473 train / 1,319 test",
    "framework": "verl / GRPO",
    "image": "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12",
    "created": "2026-11-05",
    "license": "Apache-2.0",
    "brief": "Training rollouts are nucleus-sampled at top_p=0.5 but trained as if on-policy, so the unsampled half of the distribution is never pushed down and entropy explodes.",
    "issue": "Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K. run_gsm8k_grpo.sh sets ROLLOUT_TOP_P=0.5 while verl recomputes old_log_probs over the full softmax with rollout_correction disabled, so GRPO treats the nucleus-truncated samples as on-policy: tokens outside the nucleus never receive a negative advantage while updates move their mass into the unsampled tail. actor/entropy rises 0.26 -> 5.06 (control 0.06), 227 of 1,024 training responses at step 50 derail into mixed-script garbage, 45 of 128 GRPO groups become eight byte-identical responses, and held-out strict accuracy peaks at 0.500 at step 10 and ends at 0.486 against 0.566 with top_p=1.0. No crash, RUN_EXIT 0."
  },
  {
    "id": "configuration/verl_grpo_adamw_eps_too_large",
    "title": "AdamW epsilon larger than the gradients it normalizes",
    "family": "Configuration",
    "tags": [
      "verl",
      "grpo",
      "gsm8k",
      "optimizer",
      "adamw",
      "epsilon",
      "slow-learning"
    ],
    "model": "Qwen2.5-0.5B-Instruct",
    "dataset": "GSM8K (openai/gsm8k main): 7,473 train / 1,319 test",
    "framework": "verl / GRPO",
    "image": "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12",
    "created": "2026-11-05",
    "license": "Apache-2.0",
    "brief": "An AdamW eps override of 1e-3 sits far above this model's per-parameter gradients, turning every update into momentum SGD 10-100x smaller than the learning rate implies.",
    "issue": "Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K. run_gsm8k_grpo.sh overrides the actor AdamW eps to 1e-3; per-parameter RMS gradients of this 0.5B model are about 1e-5 to 5e-5, so eps dominates the Adam denominator and the update becomes momentum SGD 10-100x smaller than lr 2e-6 implies: pg_clipfrac averages 8e-5 over steps 1-10 (control 2e-3), the training score first reaches 0.3 at step 35 (control step 5), and held-out strict accuracy ends at 0.486 against 0.566 with the default eps, while every per-step metric looks stable. No crash, RUN_EXIT 0."
  },
  {
    "id": "algorithm/maestro_grpo_reward_collapse",
    "title": "Search-token reconstruction drops the end token and collapses multimodal GRPO",
    "family": "Algorithm",
    "tags": [
      "maestro",
      "verl",
      "grpo",
      "multimodal",
      "tool-use",
      "token-truncation",
      "issue-only"
    ],
    "model": "Qwen/Qwen3-VL-4B-Thinking (+ six staged tool-service models)",
    "dataset": "locaszzz/maestro_data: 9,200 train / 100 held-out",
    "framework": "verl + verl_tool (Maestro) / GRPO",
    "image": "maestro-vllm011",
    "created": "2026-11-05",
    "license": "MIT",
    "brief": "The first-turn search-action reconstruction omits <|im_end|> before the final-token replacement eats the closing '>' of </search>, so recorded rollouts carry corrupted delimiters and training collapses.",
    "issue": "Training Qwen3-VL-4B-Thinking with multimodal GRPO over the Maestro tool loop (six staged tool-service models, 200 steps). The first-turn <search>...</search> reconstruction appends no <|im_end|>, and the new_gen_tokens[:-1] + gen_ids[-1:] final-token replacement assumes the reconstructed last token is the EOS and instead removes the '>' that closes </search>: sampled step-1 rollouts carry unequal opening/closing search-tag counts with closing tags like </search<|im_end|>, degenerating to forms like </search<<|im_start|> by step 200. At step 200 critic/score/mean=-1.0, verl_tool/is_traj_finished/mean=0.0, and responses clip at the 4096-token cap (response_length/clip_ratio=0.5625, response_length/max=4096) with no completed final-answer trajectory among the sampled rows. Training reward still improves temporarily (step-120 mean 0.71), so never having learned is a wrong attribution. No crash."
  },
  {
    "id": "infrastructure/r1d15_math_grpo",
    "title": "Numerical policy divergence in a completed math GRPO run",
    "family": "Infrastructure",
    "tags": [
      "verl",
      "grpo",
      "sequence-mis",
      "numerical-consistency",
      "hybrid-backend"
    ],
    "model": "DeepSeek-R1-Distill-Qwen-1.5B",
    "dataset": "MATH 1460-question subset (train); AIME 2024 / 2025 (held-out)",
    "framework": "verl / GRPO + sequence-MIS",
    "image": "verl-vllm0.8.5-torch2.6.0-flashattn2.7.4-cu124-py3.10",
    "created": "2026-11-06",
    "license": "Apache-2.0",
    "brief": "DeepSeek-R1-Distill-Qwen-1.5B trains with GRPO + sequence-MIS and the run completes with improving AIME, but the BF16 FSDP2 learner forward and the vLLM rollout forward compute materially different token probabilities for the same synchronized weights.",
    "issue": "Training DeepSeek-R1-Distill-Qwen-1.5B with GRPO + sequence-MIS on the MATH 1460-question subset (440 steps). With both forwards at bfloat16, the recorded token-weighted mean absolute logprob difference is 0.0112137 nat/token over 1.03 billion response tokens, the maximum sampled-token probability difference reaches 0.999926 (step 252), the max absolute logprob difference is 17.5485 (step 326), the mean sequence ESS fraction collapses to 0.0989, and 5.65% of sequences are mis-rejected by the sequence log-ratio gate. An FP16 arm of the same experiment measures 0.00135637 nat/token with ESS 0.9506 -- an 8.27x separation -- while both arms learn and finish with finite optimizer records, and BF16 finishes within 2.81 / 2.71 points of FP16 on AIME 2024 / 2025 avg@32: the run looks healthy and is not."
  },
  {
    "id": "data/qwen25_gsm8k_grpo",
    "title": "Qwen2.5 GSM8K GRPO run",
    "family": "Data",
    "tags": [
      "input-contamination",
      "data-preprocessing",
      "issue-only"
    ],
    "model": "Qwen2.5-1.5B-Instruct",
    "dataset": "GSM8K (7,473 train / 1,319 test; 256-row held-out validation subset)",
    "framework": "verl / GRPO",
    "image": "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12",
    "created": "2026-11-07",
    "license": "Apache-2.0",
    "brief": "A preprocessing regression appends each selected training row's own gold answer to about 20% of prompts; nothing crashes and the run improves, but reward flows to copying the disclosed answer.",
    "issue": "Training Qwen2.5-1.5B-Instruct with GRPO on GSM8K completes 40 steps. The preprocessor appends the same row's gold answer to 1,494 of 7,473 training prompts; all labels, row order and 1,319 test rows remain unchanged. The affected subset's online accuracy rises from 9.5% at step 1 to 96.0% at step 40; a recorded rollout computes 55/6 and then explicitly adopts the reference answer; clean test accuracy reaches 78.09%. The native scorer is correct -- the run looks healthy and is not."
  },
  {
    "id": "data/verl_taco_wrong_expected_output",
    "title": "Native wrong expected outputs in the training labels silence correct solutions",
    "family": "Data",
    "tags": [
      "verl",
      "grpo",
      "taco",
      "label-noise",
      "false-negative",
      "data-quality"
    ],
    "model": "Qwen3-4B-Instruct-2507",
    "dataset": "TACO train mix (~1,380 rows, ~40% natively wrong-expected-output dirty pool, ~828 verified-clean) / TACO-verified EASY 184-row held-out",
    "framework": "verl / GRPO",
    "image": "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12",
    "created": "2026-11-06",
    "license": "MIT",
    "brief": "About 40% of the TACO training rows are problems whose expected outputs are natively wrong, so a correctly solved rollout still scores 0 and GRPO teaches the model to discard correct algorithms for those problems.",
    "issue": "Training Qwen3-4B-Instruct-2507 with GRPO on a matched TACO difficulty mix. The problems' own bundled solutions agree on outputs that still fail their own tests; under the binary all-tests-pass reward, a rollout that solves such a problem correctly is scored 0. Training reward stays pinned below the clean baseline scaled by the clean fraction, the correct-but-scored-0 rate persists instead of vanishing, and held-out accuracy regresses instead of improving."
  },
  {
    "id": "configuration/areal_icepop_mask_window_too_narrow",
    "title": "IcePop mask window narrower than the train-inference ratio spread",
    "family": "Configuration",
    "tags": [
      "areal",
      "token-masking",
      "importance-weight-window",
      "train-inference-mismatch",
      "entropy-blowup",
      "delayed-collapse"
    ],
    "model": "Qwen1.5-MoE-A2.7B-Chat",
    "dataset": "GSM8K (openai/gsm8k main): 7,473 train / 1,319 test",
    "framework": "AReaL / decoupled PPO with IcePop token masking",
    "image": "areal-v2.0.0-sglang0.5.10.post1-torch2.9.1-flashattn2.8.3-cu129-py3.12",
    "created": "2026-11-10",
    "license": "Apache-2.0",
    "brief": "run.sh narrows the IcePop token-mask window from the shipped [0.5, 5] to [0.67, 1.5]; the masked token fraction starts 4x higher and the run collapses while nothing crashes.",
    "issue": "Training Qwen1.5-MoE-A2.7B-Chat with decoupled PPO and IcePop token masking. run.sh narrows the mask window (actor.rejection_sampling.lower/upper) from the shipped [0.5, 5] to [0.67, 1.5]; on this MoE model the behaviour importance weight spreads wider than that, so the masked token fraction starts 4x higher (0.0161 vs 0.0041), crosses 0.10 at step 43 and peaks at 0.750 at step 73 while entropy rises from 0.184 to 5.92 and responses lengthen from 284 to 600 tokens; held-out GSM8K eval-rollout/reward regresses against the shipped-window control."
  },
  {
    "id": "infrastructure/areal_disk_weight_update_on_sglang",
    "title": "Disk weight sync inherited from a vLLM workaround stalls an SGLang run",
    "family": "Infrastructure",
    "tags": [
      "areal",
      "decoupled-ppo",
      "gsm8k",
      "weight-sync",
      "throughput",
      "config-drift",
      "moe",
      "issue-only"
    ],
    "model": "Qwen1.5-MoE-A2.7B-Chat",
    "dataset": "GSM8K (openai/gsm8k main): 7,473 train / 1,319 test",
    "framework": "AReaL / decoupled PPO with truncated importance sampling (TIS)",
    "image": "areal-v2.0.0-sglang0.5.10.post1-torch2.9.1-flashattn2.8.3-cu129-py3.12",
    "created": "2026-11-10",
    "license": "Apache-2.0",
    "brief": "The shared example YAML switches weight sync to disk to work around a vLLM hang; on this SGLang run every step writes and reloads the full ~28 GB model, and the run is cut off by its 4-hour limit after 65 of 87 steps while still learning.",
    "issue": "Training Qwen1.5-MoE-A2.7B-Chat on GSM8K with asynchronous decoupled PPO (TIS) in AReaL with SGLang rollouts. The shared YAML switched weight sync to disk to work around a vLLM hang, so every step writes and reloads the full ~28 GB model: median update_weights is 144.4 s against 3.6 s with NCCL (xccl), each step takes 172.5 s instead of 35.0 s, and the 87-step run is cut off by its 4-hour limit after 65 steps. Held-out eval-rollout/reward (0.6304, 0.6734) tracks the xccl control (0.6674, 0.6721) -- the run looks healthy and is not."
  }
];
