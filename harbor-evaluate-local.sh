# Logs land in outputs/harbor-evaluate/<task_id>/<agent>-<model>/<timestamp>/ automatically.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://yeysai.com/v1}"

python3 -m autorlbench harbor-evaluate tasks/algorithm/verl_grpo_advantage_broadcast \
  --evaluation-level solution \
  --judge-model gpt-5.5 \
  --huggingface-mirror https://hf-mirror.com \
  --agent codex --model gpt-5.5
