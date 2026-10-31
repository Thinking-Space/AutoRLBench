"""Task packaging contracts, using only pinned public files and local fixtures."""
import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/data/justrl2_data_relocation'


class TaskPackaging(unittest.TestCase):
    def test_prepare_exposes_pinned_framework_source_to_diagnosis_agent(self):
        source = (TASK / 'workspace/prepare_assets.py').read_text()
        names = {node.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)}
        self.assertIn('expose_framework_source', names)
        self.assertIn('expose_framework_source', (TASK / 'environment/prepare.sh').read_text())

    def test_reference_repair_is_hidden_and_single_mode_has_no_repair_branch(self):
        storage = (TASK / 'workspace/runtime/storage.py').read_text()
        names = {n.name for n in ast.walk(ast.parse(storage)) if isinstance(n, ast.FunctionDef)}
        self.assertNotIn('_repaired_bytes', names)
        self.assertFalse((TASK / 'workspace/verify.py').exists())
        self.assertFalse((TASK / 'workspace/evaluations').exists())
        self.assertTrue((TASK / 'evaluations/solution/oracle/repair.patch').exists())

    def test_canonical_scaffolds_and_image_tooling_are_preserved(self):
        canonical = ROOT / 'tasks/reward/slime_retool_length_reward_bonus/evaluations'
        for name in ['instruction.md', 'verifier/test.sh', 'verifier/judge.py', 'verifier/judge_prompt.md']:
            self.assertEqual((TASK / 'evaluations/issue' / name).read_bytes(), (canonical / 'issue' / name).read_bytes())
        self.assertEqual((TASK / 'evaluations/solution/verifier/test.sh').read_bytes(),
                         (canonical / 'solution/verifier/test.sh').read_bytes())
        marker = '# ---- AutoRLBench tooling: keep identical in every docker/images/*.Dockerfile ----'
        actual = (ROOT / 'docker/images/justrl2-data-relocation.Dockerfile').read_text().split(marker)[1]
        expected = (ROOT / 'docker/images/verl.Dockerfile').read_text().split(marker)[1]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
