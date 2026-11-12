#!/usr/bin/env bash
# Issue-level sweep: run one agent+model over every AutoRLBench task.
#
#   bash skills/harbor-evaluate/examples/run-issue.sh <agent> <model> [judge-model] [task...]
#
#   <agent>        e.g. claude-code, codex, qwen-coder, grok-build
#   <model>        the model id the agent will serve
#   [judge-model]  issue judge (OpenAI-compatible, via OPENAI_BASE_URL/OPENAI_API_KEY);
#                  defaults to $AUTORLBENCH_JUDGE_MODEL
#   [task...]      optional explicit task paths; default: every task under tasks/
#
# Each task is its own harbor-evaluate invocation, so one failure does not stop
# the sweep, and each run's command.sh can be replayed alone. Runs land under
# --output-root (default outputs/eval-issue-<agent>-<model>), isolated from
# other sweeps. The process exit code is NOT the score: summarize afterwards
# with skills/harbor-evaluate/examples/collect-results.py, which applies the validity rules.
set -uo pipefail
cd "$(dirname "$0")/../../.."

agent="${1:?agent, e.g. claude-code}"
model="${2:?model id}"
judge="${3:-${AUTORLBENCH_JUDGE_MODEL:-}}"
[ -n "$judge" ] || { echo "pass a judge model (arg 3) or set AUTORLBENCH_JUDGE_MODEL" >&2; exit 2; }
shift 2
[ $# -gt 0 ] && [ "${1:-}" != "$judge" ] || shift $(( $# > 0 ? 1 : 0 ))

# Discover tasks: children of each domain directory that carry a task.toml.
if [ $# -gt 0 ]; then
  tasks=("$@")
else
  tasks=()
  for domain in tasks/*/; do
    for task in "$domain"*/; do
      [ -f "${task}task.toml" ] && tasks+=("${task%/}")
    done
  done
fi
[ "${#tasks[@]}" -gt 0 ] || { echo "no tasks found under tasks/" >&2; exit 2; }

label="$(printf '%s-%s' "$agent" "$model" | tr -c 'A-Za-z0-9._-' '-')"
OUT="outputs/eval-issue-${label}"
echo "sweep: ${#tasks[@]} tasks | agent=$agent model=$model judge=$judge"
echo "output: $OUT"
mkdir -p "$OUT"

for t in "${tasks[@]}"; do
  echo "=============== issue $(basename "$t") [$agent/$model] $(date '+%F %T') ==============="
  python3 -m autorlbench harbor-evaluate "$t" \
    --evaluation-level issue \
    --judge-model "$judge" \
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
