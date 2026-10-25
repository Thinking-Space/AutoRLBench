"""Current evidence integrity and replay; no external maintenance backups required."""
from collections import Counter
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'task.toml').is_file())
TOOLS = ROOT / 'evaluations/issue/verifier/resources/tools'
sys.path.insert(0, str(TOOLS))
import curate_reward_trajectories as replay
import probe_dashboard_verifier as dashboard

EXPECTED_STEPS = {1: 4, 150: 4, 175: 3, 190: 2, 199: 2, 200: 5}
EXPECTED_KINDS = {'dashboard': 1, 'shipment_runtime': 2, 'watchlist_runtime': 1,
                  'firmware_no_action': 1, 'device_count': 1, 'leaderboard': 1,
                  'thread_no_action': 1}

class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / 'evaluations/issue/verifier/resources/curated_trajectory_manifest.json').read_text())

    def test_current_counts_and_complete_aligned_episodes(self):
        counts = self.manifest['counts']
        self.assertEqual(counts['unique_episodes'], 20)
        self.assertEqual(counts['confirmed_reward_errors'], 8)
        self.assertEqual(counts['other_context_episodes'], 12)
        self.assertEqual(counts['by_step'], {str(k): v for k, v in EXPECTED_STEPS.items()})
        self.assertEqual({int(p.stem) for p in replay.evidence_paths(ROOT / 'workspace/logs/rollouts')}, set(EXPECTED_STEPS))
        seen = set()
        for step, count in EXPECTED_STEPS.items():
            episodes = replay.records(ROOT / f'workspace/logs/rollouts/{step}.jsonl')
            analyses = replay.records(ROOT / f'workspace/logs/idea_analysis/step_{step:08d}.jsonl')
            self.assertEqual(len(episodes), count)
            self.assertEqual(set(episodes), set(analyses))
            for uid, rows in episodes.items():
                indices = [row[2]['step_num'] for row in rows]
                self.assertEqual(sorted(set(indices)), list(range(max(indices) + 1)))
                self.assertEqual(indices, sorted(indices))
                self.assertNotIn(uid, seen)
                seen.add(uid)
        audited = {case['traj_uid'] for case in self.manifest['cases']}
        self.assertEqual(len(audited), 8)
        self.assertLessEqual(audited, seen)
        oracle = (ROOT / 'evaluations/issue/oracle.md').read_text()
        self.assertLessEqual(set(re.findall(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', oracle)), seen)

    def test_current_artifact_and_episode_hashes(self):
        for item in self.manifest['retained_files']:
            self.assertEqual(replay.artifact(ROOT, ROOT / item['path']), item)
        for case in self.manifest['cases']:
            for item in case['artifacts']:
                self.assertEqual(replay.artifact(ROOT, ROOT / item['path']), item)
            self.assertTrue(replay.episode_records(ROOT, case))

    def test_eight_reward_counterexamples_replay(self):
        result = replay.replay(ROOT)
        self.assertEqual(result['episodes'], 8)
        self.assertEqual(Counter(x['kind'] for x in result['cases']), EXPECTED_KINDS)
        self.assertEqual(Counter(x['evidence_level'] for x in result['cases']),
                         {'archived_database_and_history': 1, 'recorded_history_and_seed_probe': 7})
        self.assertTrue(all(x['semantic_contradiction_confirmed'] for x in result['cases']))

    def test_seed_fixtures_are_not_claimed_as_historical_databases(self):
        for case in self.manifest['cases']:
            audit = json.loads((ROOT / case['audit_path']).read_text())
            if case['evidence_level'] == 'archived_database_and_history':
                self.assertTrue(audit['source_databases'])
                self.assertIn('initial_tables', audit)
                self.assertIn('final_tables', audit)
            else:
                self.assertEqual(audit['source_databases'], [])
                self.assertIn('NOT an archived historical', audit['fixture_scope'])
                self.assertTrue(audit['seed_fixture_tables'])

    def test_dashboard_counterfactual(self):
        cases = dashboard.replay(ROOT)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]['no_actions'], {'result': 'complete'})
        self.assertEqual(cases[0]['incomplete_agent_dashboard'], {'result': 'complete'})
        self.assertEqual(cases[0]['remove_other_users_seeded_widgets'], {'result': 'others'})

if __name__ == '__main__':
    unittest.main()
