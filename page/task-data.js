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
      "aime"
    ],
    "model": "Qwen3-4B",
    "dataset": "DAPO-Math-17k (RL) + ReTool-SFT (SFT)",
    "framework": "slime",
    "image": "slime-v0.3.2",
    "created": "2026-08-19",
    "license": "Apache-2.0",
    "brief": "Model hacks the reward by generating longer answers.",
    "issue": "Training Qwen3-4B with GRPO on DAPO-Math-17k. The model hacks the reward by generating longer answers."
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
    "id": "environment/alfworld_tool_schema_mismatch",
    "title": "ALFWorld tool-schema name mismatch",
    "family": "Environment",
    "tags": [
      "agent-r1",
      "grpo",
      "alfworld",
      "tool-schema"
    ],
    "model": "Qwen/Qwen3-4B",
    "dataset": "ALFWorld difficulty-v1-s42 train (1008 draws) + balanced1008-s42 validation",
    "framework": "agent-r1",
    "image": "verl",
    "created": "2026-10-20",
    "license": "MIT",
    "brief": "Renamed tool calls are never accepted by the executor.",
    "issue": "Training Qwen3-4B with GRPO on ALFWorld. The tool schema renames clean/heat/cool to clean_object/heat_object/cool_object, but the executor registry only knows the original names, so those action families stay at zero success."
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
    "id": "reward/proxy_reward_misalignment",
    "title": "Brevity proxy reward misalignment",
    "family": "Reward",
    "tags": [
      "verl",
      "ppo",
      "proxy-reward",
      "gurumath"
    ],
    "model": "Qwen3-4B-Base",
    "dataset": "GuruMath fixed-16 controlled set",
    "framework": "verl",
    "image": "vimpo-cu128-recorded",
    "created": "2026-08-18",
    "license": "MIT",
    "brief": "A brevity proxy optimizes for short, worse answers.",
    "issue": "Training Qwen3-4B-Base with PPO on GuruMath. The reward uses a brevity proxy (1 - n_chars/4000), so training reward climbs while held-out math accuracy falls from 0.781 to 0.594 and responses shorten from ~677 to ~474 tokens."
  },
  {
    "id": "reward/reward_parser_mismatch",
    "title": "Reward parser vs ground-truth format mismatch",
    "family": "Reward",
    "tags": [
      "verl",
      "ppo",
      "reward-parsing",
      "gurumath"
    ],
    "model": "Qwen3-4B-Base",
    "dataset": "GuruMath controlled smoke (1024 rows)",
    "framework": "verl",
    "image": "ppo-critic-vimpo",
    "created": "2026-08-18",
    "license": "MIT",
    "brief": "The parser never matches, so every reward is zero.",
    "issue": "Training Qwen3-4B-Base with PPO on GuruMath. The reward parser extracts XML <answer> tags while ground truth is \\boxed{} format, so both training reward and validation accuracy stay at exactly zero for the whole run."
  },
  {
    "id": "reward/slime_search_r1_nomask",
    "title": "Search-R1 observations trained without a loss mask",
    "family": "Reward",
    "tags": [
      "slime",
      "grpo",
      "search-r1",
      "loss-mask"
    ],
    "model": "Qwen2.5-3B",
    "dataset": "Search-R1 nq_hotpotqa_train",
    "framework": "slime",
    "image": "slime-v0.3.2",
    "created": "2026-08-19",
    "license": "Apache-2.0",
    "brief": "Retrieved documents get gradient, so the model stops searching.",
    "issue": "Training Qwen2.5-3B with GRPO on Search-R1 NQ/HotpotQA. Retrieved <information> tokens are trained (loss_mask=1), so response length collapses from ~1751 to single digits and NQ test accuracy lands at 16.5% versus a 40.8% baseline."
  },
  {
    "id": "reward/idea_awm_reward_diagnosis",
    "title": "Inaccurate rule-based task rewards in IDEA AWM OPSD",
    "family": "Reward",
    "tags": [
      "idea",
      "verl",
      "awm",
      "opsd",
      "issue-only"
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
    "id": "reward/task_function_call",
    "title": "Function-call problems always score zero",
    "family": "Reward",
    "tags": [
      "verl",
      "grpo",
      "taco",
      "function-call"
    ],
    "model": "Qwen2.5-1.5B-Instruct",
    "dataset": "TACO EASY",
    "framework": "verl",
    "image": "verl",
    "created": "2026-08-21",
    "license": "MIT",
    "brief": "The reward ignores fn_name, deleting call-based scoring.",
    "issue": "Training Qwen2.5-1.5B-Instruct with GRPO on TACO EASY. The taco.py reward function ignores fn_name and deletes call-based execution, so every function-call problem scores exactly zero."
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
      "throughput"
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
    "id": "infrastructure/trl_conditional_reward_logging",
    "title": "Distributed GRPO with conditional reward metrics",
    "family": "Infrastructure",
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
    "brief": "A rank-0-only metric makes the two ranks issue different collectives.",
    "issue": "Training Qwen2.5-0.5B-Instruct with two-rank distributed GRPO on GSM8K. Each rank iterates its own pending metric and extra-column keys before invoking collectives, while the reward function emits answer_parse_rate only on rank 0, so the ranks call different collectives at the same sequence number and torch.distributed reports a collective fingerprint mismatch before the first optimizer step."
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
    "id": "reward/verl_grpo_gsm8k_any_number_reward",
    "title": "GSM8K reward credits the gold number anywhere in the response",
    "family": "Reward",
    "tags": [
      "verl",
      "grpo",
      "gsm8k",
      "reward-hacking",
      "lenient-verifier",
      "answer-extraction"
    ],
    "model": "Qwen2.5-0.5B-Instruct",
    "dataset": "GSM8K train 7,473 / held-out test 1,319",
    "framework": "verl / GRPO",
    "image": "verl-0.9-vllm0.28",
    "created": "2026-10-27",
    "license": "Apache-2.0",
    "brief": "A hand-written reward gives full credit whenever the gold number appears anywhere in the response, and the run climbs while the policy stops committing to any final answer.",
    "issue": "Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K. The custom reward rewards/gsm8k_reward.py — loaded through reward.custom_reward_function and also reused for the logged validation metric — returns 1.0 whenever the gold answer is among all numbers in the response, so GRPO rewards mentioning more numbers and longer enumerations instead of a committed final answer: the logged training score rises 0.54 -> 0.81 and logged validation accuracy 0.594 -> 0.684 over 100 steps, while the same held-out responses re-scored strictly are 0.000 from step 10 on and format-free accuracy falls from 0.462 to 0.130; distinct numbers per response grow 9.4 -> 14.3 and '####' disappears from the rollouts."
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
  }
];
