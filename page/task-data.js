/*
 * Public task index. Each id is the task path under tasks/ in the repository.
 * Header metadata is copied from each task.toml by
 * scripts/generate_task_data.py; `brief` and `issue` are hand-written and
 * preserved across regenerations.
 */
window.AUTORLBENCH_TASKS = [
    {
        "id": "configuration/verl_aime25_single_sample_protocol",
        "title": "AIME25 single-sample variance",
        "family": "Configuration",
        "tags": [
            "verl",
            "ppo",
            "sampling",
            "aime25"
        ],
        "model": "Qwen3-4B-Base",
        "dataset": "AIME 2025",
        "framework": "verl",
        "image": "ppo-critic-vimpo",
        "created": "2026-08-18",
        "license": "MIT",
        "brief": "One sample per question makes evaluation too noisy.",
        "issue": "Evaluating Qwen3-4B-Base on AIME 2025. One sample per question makes the score too noisy to judge the run."
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
        "created": "2026-10-10",
        "license": "MIT",
        "brief": "Training set is very hard while validation set is easy.",
        "issue": "Training Qwen2.5-1.5B-Instruct with GRPO on TACO. The training set is very hard while the validation set is easy."
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
            "uid"
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
    }
];
