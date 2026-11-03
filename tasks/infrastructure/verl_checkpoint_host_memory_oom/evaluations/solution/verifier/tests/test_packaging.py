"""Offline packaging checks: pin coverage, frozen-evidence validity, oracle diff.

Standard library only, so these run on the maintenance host without the
training image (no torch, no omegaconf). GPU grading lives in
grade_submission.py; these tests gate the packaging itself.
"""
import difflib
import hashlib
import json
import sys
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
TASK = TESTS.parents[3]
WORKSPACE = TASK/'workspace'
SPEC = json.loads((TESTS.parent/'resources/grader_spec.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PackagingTests(unittest.TestCase):

    def test_pins_match_workspace(self):
        for rel, digest in SPEC['protected_source_sha256'].items():
            file = WORKSPACE/rel
            self.assertTrue(file.is_file(), 'pinned file missing: '+rel)
            self.assertFalse(file.is_symlink(), 'pinned file replaced by symlink: '+rel)
            self.assertEqual(sha(file), digest, 'pin mismatch: '+rel)

    def test_workspace_inventory_closes(self):
        allowed = set(SPEC['protected_source_sha256']) | {'configs/experiment.yaml', 'report.md'}
        allowed |= {'model/'+asset['path'] for asset in SPEC['model_assets']}
        for path in WORKSPACE.rglob('*'):
            if path.is_dir():
                continue
            rel = path.relative_to(WORKSPACE).as_posix()
            if '__pycache__' in rel or rel.endswith('.pyc'):
                continue
            self.assertIn(rel, allowed, 'unpinned workspace file: '+rel)
        for rel in SPEC['protected_sha256']:
            self.assertIn(rel, SPEC['protected_source_sha256'])

    def test_frozen_defective_run_passes_failure_check(self):
        sys.path.insert(0, str(TESTS))
        try:
            from check_run import check_checkpoint_failure
        finally:
            sys.path.remove(str(TESTS))
        check_checkpoint_failure(WORKSPACE/'logs',
                                 (WORKSPACE/'logs/training.stderr.log').read_text(),
                                 SPEC['memory_limit_bytes'])

    def test_defective_config_differs_only_in_optimizer_offload(self):
        defective = (WORKSPACE/'configs/experiment.yaml').read_text().splitlines(keepends=True)
        repaired = (TASK/'evaluations/solution/oracle/experiment.yaml').read_text().splitlines(keepends=True)
        diff = [line for line in difflib.unified_diff(defective, repaired, n=0)
                if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))]
        self.assertEqual(len(diff), 2, 'expected exactly one changed line, got: '+repr(diff))
        self.assertIn('optimizer_offload', diff[0]+diff[1])
        self.assertIn('false', diff[1])
        self.assertIn('true', diff[0])

    def test_verifier_scripts_compile(self):
        for name in ('check_run.py', 'grade_submission.py'):
            compile((TESTS/name).read_text(), name, 'exec')


if __name__ == '__main__':
    unittest.main()
