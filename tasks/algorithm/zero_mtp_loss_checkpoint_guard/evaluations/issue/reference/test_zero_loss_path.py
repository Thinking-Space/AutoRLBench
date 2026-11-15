"""CPU integration checks on the actual preserved Bash helper."""
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3] / "workspace"
sys.path.insert(0, str(ROOT))
from runtime.checkpoint_policy import check_guard

class ActualCheckpointSourceTest(unittest.TestCase):
    def test_original_source_rejects_one_rollout_schedule(self):
        result = check_guard(ROOT / "runtime/source/checkpoint_args.sh")
        self.assertFalse(result["accepted"])

    def test_original_source_accepts_periodic_normal_schedule(self):
        result = check_guard(ROOT / "runtime/source/checkpoint_args.sh", num_rollouts=8,
                             save_interval=2, debug_train_only=False)
        self.assertTrue(result["accepted"])

    def test_preserved_fixed_source_scopes_debug_exception(self):
        fixed = ROOT.parent / "evaluations/issue/reference/checkpoint_args.sh"
        self.assertTrue(check_guard(fixed)["accepted"])
        self.assertFalse(check_guard(fixed, debug_train_only=False)["accepted"])

if __name__ == "__main__":
    unittest.main()
