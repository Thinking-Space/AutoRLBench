#!/usr/bin/env bash
# Solution-level sweep: run one agent+model over every task that supports it.
#
#   bash skills/harbor-evaluate/examples/run-solution.sh <agent> <model> [task...]
#
#   <agent>     e.g. claude-code, codex, qwen-coder, grok-build
#   <model>     the model id the agent will serve
#   [task...]   optional explicit task paths; default: every task whose
#               evaluations/solution/ exists
#
# Differences from the issue sweep: no --judge-model (the solution verifier is
# the task's own deterministic grader), and the verifier stage re-runs training
# inside the container, so tasks whose task.toml declares environment.gpus > 0
# need this machine to actually have those GPUs -- check `docker run --rm --gpus
# all nvidia-smi` first if unsure.
#
# Each task is its own harbor-evaluate invocation, so one failure does not stop
# the sweep, and each run's command.sh can be replayed alone. The process exit
# code is NOT the score: summarize with examples/collect-results.py afterwards.
set -uo pipefail
cd "$(dirname "$0")/../../.."

agent="${1:?agent, e.g. claude-code}"
model="${2:?model id}"
shift 2

# Discover solution-capable tasks: a task.toml plus evaluations/solution/.
if [ $# -gt 0 ]; then
  tasks=("$@")
else
  tasks=()
  for domain in tasks/*/; do
    for task in "$domain"*/; do
      if [ -f "${task}task.toml" ] && [ -d "${task}evaluations/solution" ]; then
        tasks+=("${task%/}")
      fi
    done
  done
fi
[ "${#tasks[@]}" -gt 0 ] || { echo "no solution-capable tasks found under tasks/" >&2; exit 2; }

if ! docker run --rm --gpus all nvidia-smi >/dev/null 2>&1; then
  echo "warning: no usable docker GPUs detected; solution tasks that rerun training will fail" >&2
fi

label="$(printf '%s-%s' "$agent" "$model" | tr -c 'A-Za-z0-9._-' '-')"
OUT="outputs/eval-solution-${label}"
echo "sweep: ${#tasks[@]} tasks | agent=$agent model=$model"
echo "output: $OUT"
mkdir -p "$OUT"

for t in "${tasks[@]}"; do
  echo "=============== solution $(basename "$t") [$agent/$model] $(date '+%F %T') ==============="
  python3 -m autorlbench harbor-evaluate "$t" \
    --evaluation-level solution \
    --output-root "$OUT" \
    --agent "$agent" --model "$model" \
    --agent-kwarg reasoning_effort=max
  rc=$?
  # The newest run directory under this sweep's output root is the one just finished.
  run=$(ls -td "$OUT"/*/*/*/ 2>/dev/null | head -1)
  if [ -n "$run" ] && [ -f "$run/run.log" ]; then
    grep '\[RESULT\]' "$run/run.log" | tail -1
  fi
  echo "--- $(basename "$t") exit=$rc $(date '+%F %T') ---"
done

echo "SWEEP DONE: ${#tasks[@]} tasks | agent=$agent model=$model | runs under $OUT"
echo "next: python3 skills/harbor-evaluate/examples/collect-results.py --output-root $OUT"
