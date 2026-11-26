"""Standalone contracts; optional native runs use ORIGINAL_EXPORT / NEW_EXPORT."""
import importlib.util
import unittest
from pathlib import Path

RUNNER = Path(__file__).with_name('run_replay.py')


def runner():
    assert RUNNER.is_file(), 'runner must exist'
    spec = importlib.util.spec_from_file_location('run_replay', RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Contracts(unittest.TestCase):
    def test_defaults_and_bounds(self):
        r = runner()
        c = r._configuration(2, 2, None, (11, 3), 'ones', 1729)
        self.assertEqual(c['steps'], [11, 3])
        self.assertEqual(c['order'], [0, 1, 2, 3])
        self.assertEqual(r._configuration(4, 3, None, [1, 4096] * 8, 'sparse', 0)['dp'], 3)

    def test_invalid_arguments(self):
        r = runner()
        defaults = dict(cp=2, dp=2, order=None, steps=(11, 3), mask_mode='ones', seed=1729)
        for bad in ({'cp': 3}, {'dp': 1}, {'steps': []}, {'steps': [True]},
                    {'steps': [4097]}, {'steps': [3] * 17}, {'steps': '11,,3'},
                    {'order': [1, 0, 2, 3]}, {'order': [0, 1, 1, 3]}, {'mask_mode': 'bad'}):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                r._configuration(**dict(defaults, **bad))

    def test_sorted_logical_permutation(self):
        self.assertEqual(runner()._configuration(2, 2, [0, 2, 1, 3], [3], 'ones', 0)['order'], [0, 2, 1, 3])

    def test_samples_are_deterministic_and_aligned(self):
        r = runner()
        for cp in (1, 2, 4):
            for dp in range(3):
                for length in (1, 3, 11, 4096):
                    s = r._samples(length, cp, dp, 1729, 0, 'sparse')
                    self.assertEqual(s, r._samples(length, cp, dp, 1729, 0, 'sparse'))
                    self.assertEqual(len(s), 1)
                    self.assertEqual(s[0]['total'] % (2 * cp), 0)
                    self.assertGreaterEqual(s[0]['total'] // (2 * cp), 2)
                    self.assertEqual(len(s[0]['values']), length)
                    self.assertEqual(len(s[0]['masks']), length)


if __name__ == '__main__':
    unittest.main()
