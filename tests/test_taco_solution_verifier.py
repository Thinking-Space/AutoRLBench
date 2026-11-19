"""Exercise the offline data-repair contract without running training."""

import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


VERIFIER = (
    Path(__file__).resolve().parents[1]
    / "tasks/data/verl_taco_wrong_expected_output/evaluations/solution/verifier"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "taco_solution_grader", VERIFIER / "tests/grade_submission.py"
)
grader = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(grader)


@unittest.skipIf(pa is None, "pyarrow is required for parquet integrity checks")
class TacoSolutionIntegrityTests(unittest.TestCase):
    def setUp(self):
        fixture = tempfile.TemporaryDirectory()
        self.addCleanup(fixture.cleanup)
        self.ws = Path(fixture.name)
        self.spec = copy.deepcopy(grader.load_spec(VERIFIER / "resources/grader_spec.json"))
        for relative, resource in (
            ("examples/grpo_trainer/run_qwen3_4b_instruct_2507_taco_fsdp.sh",
             "run_qwen3_4b_instruct_2507_taco_fsdp.sh"),
            ("verl/utils/reward_score/taco.py", "taco.py"),
        ):
            destination = self.ws / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(VERIFIER / "resources" / resource, destination)
        (self.ws / "report.md").write_text("Filtered invalid training labels.\n")
        (self.ws / "taco_verl").mkdir()
        test_path = self.ws / "taco_verl/test.parquet"
        pq.write_table(pa.Table.from_pylist([self.row("held-out")]), test_path)
        # Only the held-out fixture differs from the task's pinned artifacts.
        self.spec["pinned_sha256"]["taco_verl/test.parquet"] = hashlib.sha256(
            test_path.read_bytes()
        ).hexdigest()
        env = patch.dict("os.environ", {
            "AUTORLBENCH_WORKSPACE": str(self.ws),
            "AUTORLBENCH_VERIFIER": str(VERIFIER),
        })
        env.start()
        self.addCleanup(env.stop)

    @staticmethod
    def row(question, expected="2", stdin="1\n"):
        return {
            "prompt": [{
                "role": "user",
                "content": grader.PREFIX + question + grader.STDIN_SUFFIX,
            }],
            "reward_model": {
                "ground_truth": json.dumps({"inputs": [stdin], "outputs": [expected]}),
            },
        }

    def rows(self, distinct=828, total=1200):
        return [self.row(f"problem {i % distinct}") for i in range(total)]

    def check_rows(self, rows, corrected=None):
        pq.write_table(pa.Table.from_pylist(rows), self.ws / "taco_verl/train.parquet")
        return grader.stage_integrity(self.spec, corrected or {})

    def test_filter_then_repeat_clean_rows_is_accepted(self):
        passed, details = self.check_rows(self.rows())
        self.assertTrue(passed, details)
        self.assertEqual(details["train_rows"], 1200)
        self.assertEqual(details["train_distinct_prompts"], 828)

    def test_declared_diversity_boundary_is_accepted(self):
        passed, details = self.check_rows(self.rows(distinct=800))
        self.assertTrue(passed, details)

    def test_repetition_does_not_bypass_diversity_floor(self):
        passed, details = self.check_rows(self.rows(distinct=799))
        self.assertFalse(passed)
        self.assertTrue(details["train_rows_ok"])
        self.assertFalse(details["train_distinct_prompts_ok"])

    def test_filtering_still_requires_total_row_floor(self):
        passed, details = self.check_rows(self.rows(total=1199))
        self.assertFalse(passed)
        self.assertFalse(details["train_rows_ok"])
        self.assertTrue(details["train_distinct_prompts_ok"])

    def test_repeated_wrong_labels_still_fail(self):
        rows = self.rows()
        rows[-9:] = [self.row("bad label", expected="4", stdin="2\n")] * 9
        corrected = {"bad": {
            "name": "bad label",
            "fail_index": 0,
            "fail_input": "2\n",
            "expected_sha256": grader.canonical_sha("4"),
        }}
        passed, details = self.check_rows(rows, corrected)
        self.assertFalse(passed)
        self.assertTrue(details["train_rows_ok"])
        self.assertTrue(details["train_distinct_prompts_ok"])
        self.assertEqual(details["train_residual_dirty_rows"], 9)
        self.assertFalse(details["train_residual_dirty_ok"])

    def test_held_out_overlap_still_fails(self):
        rows = self.rows()
        rows[-1] = self.row("held-out")
        passed, details = self.check_rows(rows)
        self.assertFalse(passed)
        self.assertFalse(details["train_test_disjoint"])

    def test_prompt_blinding_still_fails(self):
        rows = self.rows()
        rows[-1]["prompt"][0]["content"] = "rewritten prompt without the shipped template"
        passed, details = self.check_rows(rows)
        self.assertFalse(passed)
        self.assertFalse(details["train_prompt_template_ok"])


if __name__ == "__main__":
    unittest.main()
