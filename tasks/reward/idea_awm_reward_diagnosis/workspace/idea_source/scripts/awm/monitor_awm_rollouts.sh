#!/usr/bin/env bash
set -u
OUT="${1:-/workspace/idea_source/outputs/awm_qwen3_4b_rollouts}"
MANIFEST="${2:-/workspace/idea_source/outputs/awm_agentworld_1000_tasks_manifest_x8.jsonl}"
while true; do
  clear
  total=$(wc -l < "$MANIFEST" 2>/dev/null || echo 0)
  done_count=$(wc -l < "$OUT/baseline_rollouts.jsonl" 2>/dev/null || echo 0)
  success=$(python3 - "$OUT/baseline_rollouts.jsonl" <<'PY'
import json,sys
n=0
try:
  for line in open(sys.argv[1], encoding='utf-8'):
    try: n += int(bool(json.loads(line).get('success')))
    except Exception: pass
except FileNotFoundError: pass
print(n)
PY
)
  workers=$(pgrep -fc 'awm agent' 2>/dev/null || true)
  runner=$(pgrep -fc 'run_awm_rollouts.py' 2>/dev/null || true)
  echo "AWM rollout monitor  $(date '+%F %T UTC')"
  echo "completed: $done_count / $total   remaining: $((total-done_count))   success: $success"
  echo "runner processes: $runner   agent workers: $workers"
  echo
  free -h | sed -n '1,3p'
  echo
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  echo
  echo "Latest runner events:"
  tail -5 "$OUT/runner.log" 2>/dev/null || true
  sleep 5
done
