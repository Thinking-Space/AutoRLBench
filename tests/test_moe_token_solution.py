"""Replay real paired logs and check final-step scoring boundaries."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/infrastructure/verl_moe_token_normalization"
GRADER = TASK / "evaluations/solution/verifier/tests/grade_submission.py"
KEY = "val-core/openai/gsm8k/acc/mean@1"


class MoETokenSolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.results = Path(self.temp.name)
        self.logs = self.results / "experiment_logs"
        self.logs.mkdir()

    def run_grader(self):
        process = subprocess.run(
            [sys.executable, "-B", str(GRADER), "--results", str(self.results)],
            capture_output=True, text=True, check=False,
        )
        result = json.loads((self.results / "result.json").read_text(encoding="utf-8"))
        experiment = json.loads((self.results / "experiment_metrics.json").read_text(encoding="utf-8"))
        return process.returncode, result, experiment

    def write_log(self, text):
        (self.logs / "train.log").write_text(text, encoding="utf-8")

    def test_real_defective_console_fails_at_actual_final_step(self):
        shutil.copyfile(TASK / "workspace/observed_run/stdout.log", self.logs / "train.log")
        code, result, experiment = self.run_grader()
        self.assertEqual(code, 1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["final_score"], 0)
        self.assertEqual([p["step"] for p in experiment["metrics"]["eval_points"]], [100, 150])

    def test_real_repaired_console_passes_at_actual_final_step(self):
        shutil.copyfile(TASK / "evaluations/solution/oracle/control/stdout.log", self.logs / "train.log")
        code, result, experiment = self.run_grader()
        self.assertEqual(code, 0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["metrics"]["final_score"], 0.546875)
        self.assertEqual([p["step"] for p in experiment["metrics"]["eval_points"]], [100, 150])

    def test_validation_does_not_attach_to_an_earlier_training_line(self):
        self.write_log(f"step:149 - actor/loss:0.1\nstep:150 - {KEY}:0.546875\n")
        code, result, experiment = self.run_grader()
        self.assertEqual(code, 0)
        self.assertEqual([p["step"] for p in experiment["metrics"]["eval_points"]], [150])

    def test_step_zero_or_step_100_only_cannot_pass(self):
        for step in [0, 100]:
            with self.subTest(step=step):
                self.write_log(f"step:{step} - {KEY}:0.9\n")
                code, result, _ = self.run_grader()
                self.assertEqual(code, 1)
                self.assertFalse(result["passed"])
                self.assertIn("step 150", result["metrics"]["errors"][0])

    def test_earlier_best_cannot_replace_final_score(self):
        self.write_log(f"step:100 - {KEY}:0.9\nstep:150 - {KEY}:0.1\n")
        code, result, _ = self.run_grader()
        self.assertEqual(code, 1)
        self.assertEqual(result["metrics"]["final_score"], 0.1)

    def test_threshold_boundary_and_invalid_accuracy(self):
        for token, expected_code in [("0.4", 0), ("0.399", 1), ("1.0", 0), ("1.01", 1), ("nan", 1), ("inf", 1), ("-0.1", 1)]:
            with self.subTest(token=token):
                self.write_log(f"step:150 - {KEY}:{token}\n")
                code, _, _ = self.run_grader()
                self.assertEqual(code, expected_code)

    def test_frozen_history_outside_experiment_logs_is_ignored(self):
        (self.results / "historical.log").write_text(f"step:150 - {KEY}:0.9\n", encoding="utf-8")
        code, result, _ = self.run_grader()
        self.assertEqual(code, 1)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
