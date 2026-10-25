"""The machine-readable half of the solution rerun protocol: loading and expanding run_spec.json.

The protocol itself is described in autorlbench/evaluation/run_spec.py's module docstring. Two
things are watched here:

- load_run_spec's validation: a broken spec has to blow up at load time, with a RunSpecError that
  points at the offending field, rather than halfway into a training run;
- expand_env's expansion: $AUTORLBENCH_* references become this backend's real paths, and a
  reference to something that is not there is left exactly as written.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autorlbench.evaluation.run_spec import (
    REPORT_FILENAME,
    RUN_SPEC_FILENAME,
    RunSpec,
    RunSpecError,
    load_run_spec,
)


def write_spec(root: Path, payload) -> Path:
    path = root / "verifier" / "resources" / RUN_SPEC_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


VALID = {
    "schema_version": 1,
    "command": 'bash run.sh > "$AUTORLBENCH_RESULTS/experiment_logs/train.log"',
    "env": {"LOG_DIR": "$AUTORLBENCH_RESULTS/experiment_logs", "STEPS": "20"},
    "timeout_s": 3600,
    "log_roots": ["experiment_logs"],
}


class LoadRunSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_missing_spec_means_no_rerun(self) -> None:
        # No run_spec.json is a legal state: a pure protocol repair runs no extra experiment.
        self.assertIsNone(load_run_spec(self.root, "task_001"))

    def test_valid_spec_round_trips(self) -> None:
        write_spec(self.root, VALID)
        spec = load_run_spec(self.root, "task_001")
        self.assertEqual(spec.task_id, "task_001")
        self.assertEqual(spec.command, VALID["command"])
        self.assertEqual(spec.env, VALID["env"])
        self.assertEqual(spec.timeout_s, 3600)
        self.assertEqual(spec.log_roots, ["experiment_logs"])

    def test_env_and_log_roots_have_defaults(self) -> None:
        write_spec(self.root, {"schema_version": 1, "command": "bash run.sh"})
        spec = load_run_spec(self.root, "task_001")
        self.assertEqual(spec.env, {})
        self.assertEqual(spec.log_roots, ["experiment_logs"])
        self.assertIsNone(spec.timeout_s)

    def test_env_scalars_are_stringified(self) -> None:
        # Bare numbers are common in JSON/YAML; the spec normalizes them to str.
        write_spec(
            self.root,
            {"schema_version": 1, "command": "bash run.sh", "env": {"STEPS": 20}},
        )
        spec = load_run_spec(self.root, "task_001")
        self.assertEqual(spec.env, {"STEPS": "20"})

    def test_bad_payload_raises_run_spec_error(self) -> None:
        cases = {
            "bad schema_version": {"schema_version": 2, "command": "bash run.sh"},
            "empty command": {"schema_version": 1, "command": "  "},
            "command not a string": {"schema_version": 1, "command": ["bash"]},
            "env not an object": {"schema_version": 1, "command": "x", "env": []},
            "env value not a scalar": {
                "schema_version": 1,
                "command": "x",
                "env": {"STEPS": {"n": 1}},
            },
            "env key empty": {"schema_version": 1, "command": "x", "env": {"": "1"}},
            "env declares a harness variable": {
                "schema_version": 1,
                "command": "x",
                "env": {"AUTORLBENCH_RESULTS": "/results"},
            },
            "log_roots not a list": {
                "schema_version": 1,
                "command": "x",
                "log_roots": "experiment_logs",
            },
            "log_roots has an empty entry": {
                "schema_version": 1,
                "command": "x",
                "log_roots": ["experiment_logs", ""],
            },
            "timeout_s not positive": {
                "schema_version": 1,
                "command": "x",
                "timeout_s": 0,
            },
            "timeout_s is a bool": {
                "schema_version": 1,
                "command": "x",
                "timeout_s": True,
            },
            "top level not an object": ["not", "an", "object"],
        }
        for name, payload in cases.items():
            with self.subTest(name):
                write_spec(self.root, payload)
                with self.assertRaises(RunSpecError):
                    load_run_spec(self.root, "task_001")

    def test_broken_json_raises_run_spec_error(self) -> None:
        path = write_spec(self.root, VALID)
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(RunSpecError):
            load_run_spec(self.root, "task_001")

    def test_report_filename_is_the_protocol_contract(self) -> None:
        # instruction.md and the evaluator point at the same convention; pinned here against drift.
        self.assertEqual(REPORT_FILENAME, "report.md")


class ExpandEnvTests(unittest.TestCase):
    def _spec(self, env) -> RunSpec:
        return RunSpec(
            task_id="task_001",
            command="bash run.sh",
            env=env,
            timeout_s=None,
            log_roots=["experiment_logs"],
        )

    def test_harness_references_are_replaced(self) -> None:
        spec = self._spec({"LOG_DIR": "$AUTORLBENCH_RESULTS/experiment_logs"})
        self.assertEqual(
            spec.expand_env({"AUTORLBENCH_RESULTS": "/results"}),
            {"LOG_DIR": "/results/experiment_logs"},
        )

    def test_plain_values_pass_through_untouched(self) -> None:
        spec = self._spec({"STEPS": "20", "MODE": "offline"})
        self.assertEqual(
            spec.expand_env({"AUTORLBENCH_RESULTS": "/results"}),
            {"STEPS": "20", "MODE": "offline"},
        )

    def test_unresolvable_reference_is_left_alone(self) -> None:
        # One typo should not turn the whole evaluation into an orchestration failure.
        spec = self._spec({"A": "$AUTORLBENCH_MISSING/x", "B": "$OTHER/x"})
        self.assertEqual(
            spec.expand_env({"AUTORLBENCH_RESULTS": "/results"}),
            {"A": "$AUTORLBENCH_MISSING/x", "B": "$OTHER/x"},
        )

    def test_empty_resolved_value_keeps_the_reference(self) -> None:
        spec = self._spec({"LOG_DIR": "$AUTORLBENCH_RESULTS/experiment_logs"})
        self.assertEqual(
            spec.expand_env({"AUTORLBENCH_RESULTS": ""}),
            {"LOG_DIR": "$AUTORLBENCH_RESULTS/experiment_logs"},
        )


class RerunStageTests(unittest.TestCase):
    """What `_run_rerun` does with a spec: clear the log roots, then run it as its own stage."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("task", "visible", "workspace", "results", "environment", "prepared"):
            (self.root / name).mkdir()

    def _call(self, level: str = "solution"):
        from autorlbench.core.types import EnvironmentPlan, EvaluationConfig
        from autorlbench.evaluation.evaluator import TaskEvaluator
        from tests.test_sandbox_lifecycle import RecordingRuntime

        class CapturingRuntime(RecordingRuntime):
            def create_sandbox(self, request, *, artifact):
                self.request = request
                return super().create_sandbox(request, artifact=artifact)

            def exec(self, sandbox, spec):
                self.spec = spec
                return super().exec(sandbox, spec)

        runtime = CapturingRuntime()
        evaluator = TaskEvaluator("local")
        evaluator.runtime = runtime
        result = evaluator._run_rerun(
            type(
                "Package",
                (),
                {
                    "task_id": "task",
                    "environment": EnvironmentPlan(),
                    "paths": type(
                        "Paths", (), {"environment": self.root / "environment"}
                    )(),
                },
            )(),
            type("Evaluation", (), {"root": self.root / "task", "level": level})(),
            type(
                "Run",
                (),
                {
                    "visible_task": self.root / "visible",
                    "workspace": self.root / "workspace",
                    "results": self.root / "results",
                    "prepared": self.root / "prepared",
                    "models": self.root / "models",
                    "datasets": self.root / "datasets",
                },
            )(),
            EvaluationConfig(backend="local"),
            artifact=None,
        )
        return runtime, result

    def test_no_spec_means_no_rerun_sandbox(self) -> None:
        runtime, result = self._call()
        self.assertIsNone(result)
        self.assertEqual(runtime.events, [])

    def test_issue_level_never_reruns(self) -> None:
        write_spec(self.root / "task", VALID)
        runtime, result = self._call(level="issue")
        self.assertIsNone(result)
        self.assertEqual(runtime.events, [])

    def test_spec_drives_a_rerun_stage(self) -> None:
        write_spec(self.root / "task", VALID)
        runtime, result = self._call()
        self.assertIsNotNone(result)
        self.assertEqual(
            runtime.events, ["create:rerun", "exec:rerun", "close:rerun"]
        )
        self.assertEqual(runtime.request.stage, "rerun")
        self.assertEqual(runtime.request.workdir, "workspace")
        # The rerun must not be able to read the oracle or the grading scripts.
        self.assertIsNone(runtime.request.host_oracle)
        self.assertIsNone(runtime.request.host_verifier)
        self.assertEqual(runtime.spec.shell, VALID["command"])
        self.assertEqual(runtime.spec.timeout_s, VALID["timeout_s"])
        # $AUTORLBENCH_RESULTS became this backend's real path; a plain value passed through.
        self.assertEqual(runtime.request.env["LOG_DIR"], "/results/experiment_logs")
        self.assertEqual(runtime.request.env["STEPS"], "20")

    def test_declared_log_roots_are_cleared_first(self) -> None:
        write_spec(self.root / "task", VALID)
        stale = self.root / "results" / "experiment_logs" / "train.log"
        stale.parent.mkdir(parents=True)
        stale.write_text("a run the agent started against protocol", encoding="utf-8")
        kept = self.root / "results" / "run.log"
        kept.write_text("harness bookkeeping", encoding="utf-8")
        self._call()
        self.assertFalse(stale.parent.exists())
        self.assertTrue(kept.exists())


class ReportGateTests(unittest.TestCase):
    """No report.md, no rerun and no grading -- the solution level's submission gate."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _evaluate(self, level: str, report: str | None):
        from autorlbench.core.types import EvaluationConfig
        from autorlbench.evaluation.evaluator import TaskEvaluator
        from tests.test_harbor_evaluator import create_task
        from tests.test_sandbox_lifecycle import RecordingRuntime

        task_root = create_task(self.root)
        if report is not None:
            (task_root / "workspace" / REPORT_FILENAME).write_text(
                report, encoding="utf-8"
            )
        work_dir = self.root / "work"
        runtime = RecordingRuntime()
        evaluator = TaskEvaluator("local")
        evaluator.runtime = runtime
        result = evaluator.evaluate(
            task_root,
            EvaluationConfig(
                backend="local",
                evaluation_level=level,
                work_dir=work_dir,
                models_dir=self.root / "models",
                datasets_dir=self.root / "datasets",
                keep_artifacts=True,
            ),
        )
        run_log = next(work_dir.glob("*/results/run.log")).read_text(encoding="utf-8")
        results_dir = next(work_dir.glob("*/results"))
        return runtime, result, run_log, results_dir

    def test_missing_report_skips_rerun_and_verifier(self) -> None:
        runtime, result, run_log, _ = self._evaluate("solution", report=None)
        self.assertNotIn("exec:verifier", runtime.events)
        self.assertIn(f"[SKIP] rerun reason=no_{REPORT_FILENAME}", run_log)
        self.assertIn(f"[SKIP] verifier reason=no_{REPORT_FILENAME}", run_log)
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(
            result.metrics["failure"],
            f"agent did not submit {REPORT_FILENAME} in the workspace",
        )

    def test_empty_report_does_not_pass_the_gate(self) -> None:
        runtime, result, _, _ = self._evaluate("solution", report="")
        self.assertNotIn("exec:verifier", runtime.events)
        self.assertEqual(result.score, 0.0)

    def test_submitted_report_runs_the_verifier_and_is_archived(self) -> None:
        runtime, result, _, results_dir = self._evaluate(
            "solution", report="# Root cause\nthe thing\n"
        )
        self.assertIn("exec:verifier", runtime.events)
        self.assertNotIn("failure", result.metrics)
        self.assertEqual(
            (results_dir / REPORT_FILENAME).read_text(encoding="utf-8"),
            "# Root cause\nthe thing\n",
        )

    def test_issue_level_has_no_report_gate(self) -> None:
        runtime, result, run_log, _ = self._evaluate("issue", report=None)
        self.assertIn("exec:verifier", runtime.events)
        self.assertNotIn("[SKIP] verifier", run_log)
        self.assertNotIn("failure", result.metrics)


class ShippedRunSpecTests(unittest.TestCase):
    """Every run_spec.json in tasks/ has to load, and declare what the protocol needs."""

    def test_every_shipped_spec_loads(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        specs = sorted(repo_root.glob("tasks/*/*/evaluations/solution"))
        self.assertTrue(specs, "no solution evaluations found under tasks/")
        for evaluation_root in specs:
            with self.subTest(str(evaluation_root.relative_to(repo_root))):
                spec = load_run_spec(evaluation_root, "shipped")
                if spec is None:
                    continue
                self.assertTrue(spec.command.strip())
                # log_roots may legitimately be empty for a protocol-check task that produces
                # no experiment logs to clear (e.g. verl_aime25_single_sample_protocol).
                self.assertIsInstance(spec.log_roots, list)
                # A rerun that trains needs a real budget; falling back to the generic
                # EvaluationConfig timeout would cut a multi-hour run off at the knees.
                self.assertIsNotNone(spec.timeout_s)


if __name__ == "__main__":
    unittest.main()
