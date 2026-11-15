#!/usr/bin/env bash
# Submitter validation only. Historical logs are never read or overwritten.
set -euo pipefail
REF=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_DIR=$(cd -- "$REF/../../.." && pwd)
WORKSPACE="$TASK_DIR/workspace"
cd "$WORKSPACE"
OUT=${1:-experiment_logs/framework_pair}
case "$OUT" in /*) ;; *) OUT="$WORKSPACE/$OUT" ;; esac
set +e
timeout 360s python3 -B scripts/train_reproduction.py --steps 2 --output-dir "$OUT/baseline"
BASELINE_RC=$?
set -e
if [[ $BASELINE_RC != 1 ]]; then
  echo "Baseline must stop in the actual MTP failure; received rc=$BASELINE_RC" >&2
  exit 1
fi
timeout 360s python3 -B scripts/train_reproduction.py --steps 2   --hook-file "$REF/weight_sync_hook.py" --guard-file "$REF/checkpoint_args.sh"   --output-dir "$OUT/fixed"
python3 -B - "$OUT" <<'CHECK'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
b=json.loads((root/'baseline/probe_report.json').read_text())
f=json.loads((root/'fixed/probe_report.json').read_text())
functions={x['function'] for x in b.get('stack',[])}
assert b['exception_type']=='AttributeError' and 'clone' in b['error'] and 'NoneType' in b['error']
assert {'_postprocess','_get_embeddings','_roll_tensor_packed_seq'} <= functions
assert b['optimizer_steps']==0
assert f['phase']=='completed' and f['optimizer_steps']==2
assert f['mtp_weights_preserved'] and f['base_parameters_updated'] and f['config_restored']
assert (root/'fixed/tiny_model_optimizer.pt').is_file()
print('Real framework failure/fix pair verified; this is a new reduced smoke, not historical RL')
CHECK
