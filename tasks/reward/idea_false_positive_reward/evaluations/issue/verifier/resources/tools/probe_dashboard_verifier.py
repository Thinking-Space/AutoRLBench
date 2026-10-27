"""Counterfactual dashboard probes using the consolidated 8-case evidence."""
import json
from pathlib import Path
import sqlite3
import tempfile

import curate_reward_trajectories as base


def replay(project=base.PROJECT):
    manifest = json.loads((project / 'evaluations/issue/verifier/resources/curated_trajectory_manifest.json').read_text())
    rule = next(row for _, row in base.json_records(project / 'workspace/data/AgentWorldModel-1K/gen_verifier.pure_code.jsonl')
                if (row['scenario'], row['task_idx']) == ('task_management_16', 8))
    namespace = {}
    exec(compile(rule['verification']['code'], '<reviewed dashboard verifier>', 'exec'), namespace)
    verify = namespace['verify_task_completion']
    results = []
    for case in manifest['cases']:
        if case['kind'] != 'dashboard':
            continue
        audit = json.loads((project / case['audit_path']).read_text())
        with tempfile.TemporaryDirectory(prefix='idea-dashboard-probe-') as tmp:
            initial, final = Path(tmp) / 'initial.db', Path(tmp) / 'final.db'
            base.materialize(initial, audit['initial_tables'])
            base.materialize(final, audit['final_tables'])
            no_actions = verify(str(initial), str(initial), 'Task not performed.')
            incomplete = verify(str(initial), str(final), audit['final_answer'])
            with sqlite3.connect(final) as conn:
                conn.execute('DELETE FROM widgets WHERE dashboard_id=1')
            removed = verify(str(initial), str(final), audit['final_answer'])
            assert no_actions == incomplete == {'result': 'complete'}
            assert removed == {'result': 'others'}
            results.append({'traj_uid': case['traj_uid'], 'no_actions': no_actions,
                            'incomplete_agent_dashboard': incomplete,
                            'remove_other_users_seeded_widgets': removed})
    return results


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2))
