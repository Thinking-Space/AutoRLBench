#!/usr/bin/env python3
import json
from pathlib import Path
import re
import sys

root=Path(sys.argv[1])
outcomes=[json.loads(p.read_text()) for p in (root/'cc-trajectories').glob('*/outcome.json')]
requests=[]
for path in (root/'cc-trajectories').glob('*/request-*.json'):
    item=json.loads(path.read_text())
    if item.get('origin')=='actual_sglang_worker':requests.append(item)
log=(root/'run.log').read_text(errors='replace')
report={'trajectory_outcomes':len(outcomes),'native_websearch_http_requests':len(requests),
        'cc_websearch_tool_calls':sum(x['native_websearch_calls'] for x in outcomes),
        'tool_errors':sum(x['tool_errors'] for x in outcomes),'native_http_status_counts':{},
        'exact_match_count':sum(x['exact_match'] for x in outcomes),
        'reward_mean':sum(x['reward'] for x in outcomes)/max(1,len(outcomes)),
        'trainable_tokens':sum(x.get('trainable_tokens', 0) for x in outcomes),
        'outcomes_without_trainable_tokens':sum('trainable_tokens' not in x for x in outcomes),
        'trainable_trajectories':sum(x.get('trainable_tokens', 0) > 0 for x in outcomes),
        'checkpoint_tracker':list(map(str,(root/'checkpoints').glob('latest_checkpointed_iteration.txt'))),
        'gradient_log_lines':[line for line in log.splitlines() if re.search(r'grad_norm|grad norm|train/actor/loss|train/actor/pg_loss',line)][-20:],
        'limitations':['Short failure reproduction, not a convergence study','Native search is expected to fail; no working search provider installed',
                       'Saved checkpoint omits optimizer/RNG state and is not a full resume checkpoint']}
for item in requests:
    key=str(item['status']);report['native_http_status_counts'][key]=report['native_http_status_counts'].get(key,0)+1
(root/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
