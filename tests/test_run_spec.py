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
import shutil
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
    """Write a run_spec.json at the task root, where the loader looks for it now."""
    path = root / RUN_SPEC_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_legacy_spec(root: Path, payload) -> Path:
    """Write a run_spec.json in the pre-task-root location the loader still falls back to."""
    path = root / "evaluations" / "solution" / "verifier" / "resources" / RUN_SPEC_FILENAME
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

    def test_legacy_location_still_loads(self) -> None:
        # Tasks that have not moved their spec to the task root keep working: the loader
        # falls back to evaluations/solution/verifier/resources/run_spec.json.
        write_legacy_spec(self.root, VALID)
        spec = load_run_spec(self.root, "task_001")
        self.assertEqual(spec.command, VALID["command"])

    def test_task_root_wins_over_the_legacy_location(self) -> None:
        write_spec(self.root, VALID)
        write_legacy_spec(self.root, {"schema_version": 1, "command": "bash stale.sh"})
        spec = load_run_spec(self.root, "task_001")
        self.assertEqual(spec.command, VALID["command"])

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
                        "Paths",
                        (),
                        {
                            "root": self.root / "task",
                            "environment": self.root / "environment",
                        },
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
                    # The staged environment/ copy _prepare_run links into the run directory.
                    "environment": self.root / "environment",
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

    def test_rerun_runs_regardless_of_level(self) -> None:
        # run_spec.json is level-neutral, so the rerun stage no longer gates on the
        # evaluation level: the same experiment backs both levels.
        write_spec(self.root / "task", VALID)
        runtime, result = self._call(level="issue")
        self.assertIsNotNone(result)
        self.assertEqual(
            runtime.events, ["create:rerun", "exec:rerun", "close:rerun"]
        )

    def test_spec_drives_a_rerun_stage(self) -> None:
        write_spec(self.root / "task", VALID)
        runtime, result = self._call()
        self.assertIsNotNone(result)
        self.assertEqual(
            runtime.events, ["create:rerun", "exec:rerun", "close:rerun"]
        )
        self.assertEqual(runtime.request.stage, "rerun")
        self.assertEqual(runtime.request.workdir, "workspace")
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


class RerunCommandTests(unittest.TestCase):
    """TaskEvaluator.rerun end to end: prepare, then the experiment -- no agent, no verifier."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _rerun(
        self,
        level: str = "solution",
        report: str | None = None,
        spec=VALID,
        keep_artifacts: bool = True,
        invocation_argv: tuple[str, ...] = (),
    ):
        from autorlbench.core.types import EvaluationConfig
        from autorlbench.evaluation.evaluator import TaskEvaluator
        from tests.test_harbor_evaluator import create_task
        from tests.test_sandbox_lifecycle import RecordingRuntime

        task_root = create_task(self.root)
        if spec is not None:
            write_spec(task_root, spec)
        if report is not None:
            (task_root / "workspace" / REPORT_FILENAME).write_text(
                report, encoding="utf-8"
            )
        work_dir = self.root / "work"
        runtime = RecordingRuntime()
        evaluator = TaskEvaluator("local")
        evaluator.runtime = runtime
        result = evaluator.rerun(
            task_root,
            EvaluationConfig(
                backend="local",
                evaluation_level=level,
                work_dir=work_dir,
                models_dir=self.root / "models",
                datasets_dir=self.root / "datasets",
                keep_artifacts=keep_artifacts,
                invocation_argv=invocation_argv,
            ),
        )
        # The run log lives at <work_dir>/<task_id>/<timestamp>/, the layout rerun now
        # shares with harbor-evaluate.
        run_dir = next((work_dir / "harbor_fixture").glob("*"))
        run_log = (run_dir / "run.log").read_text(encoding="utf-8")
        return runtime, result, run_log, run_dir

    def test_reruns_the_workspace_exactly_as_shipped(self) -> None:
        # No report gate: the command exists to reproduce the buggy run, so a
        # workspace without report.md still gets its experiment rerun.
        runtime, result, run_log, _ = self._rerun(report=None)
        self.assertEqual(
            runtime.events,
            [
                "create:prepare",
                "exec:prepare",
                "close:prepare",
                "create:rerun",
                "exec:rerun",
                "close:rerun",
            ],
        )
        self.assertNotIn("[SKIP] rerun", run_log)
        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["exit_code"], 0)
        self.assertIsNone(result.score)

    def test_a_report_in_the_workspace_changes_nothing(self) -> None:
        runtime, result, _, run_dir = self._rerun(report="# Root cause\n")
        self.assertNotIn("exec:verifier", runtime.events)
        # Nothing is graded, so the report is not archived either.
        self.assertFalse((run_dir / REPORT_FILENAME).exists())
        self.assertTrue(result.passed)

    def test_no_verifier_at_any_level(self) -> None:
        runtime, result, run_log, _ = self._rerun(level="issue")
        self.assertNotIn("exec:verifier", runtime.events)
        self.assertNotIn("verifier", run_log)
        self.assertTrue(result.passed)

    def test_task_without_run_spec_is_rejected(self) -> None:
        from autorlbench.tasks import SchemaError

        with self.assertRaises(SchemaError):
            self._rerun(spec=None)

    def test_run_log_is_laid_out_and_staging_is_deleted(self) -> None:
        # The persistent run log sits at <work_dir>/<task_id>/<timestamp>/ and holds the
        # same record set as harbor-evaluate's; the digest-named staging is deleted
        # after its results and workspace are salvaged into it.
        runtime, result, _, run_dir = self._rerun(
            invocation_argv=("python3", "-m", "autorlbench", "rerun", "tasks/x")
        )
        self.assertTrue((run_dir / "run.log").is_file())
        self.assertTrue((run_dir / "report.json").is_file())
        self.assertTrue((run_dir / "results").is_dir())
        self.assertTrue((run_dir / "workspace" / "train.sh").is_file())
        command = (run_dir / "command.sh").read_text(encoding="utf-8")
        self.assertIn("python3 -m autorlbench rerun tasks/x\n", command)
        self.assertEqual(result.report_dir, run_dir)
        self.assertEqual(list(self.root.glob("work/autorlbench-run-*")), [])

    def test_workspace_is_dropped_without_keep_artifacts(self) -> None:
        # The experiment's logs always survive; the tens-of-GB workspace copy is
        # only salvaged when the caller asks for it.
        runtime, result, _, run_dir = self._rerun(keep_artifacts=False)
        self.assertFalse((run_dir / "workspace").exists())
        self.assertTrue((run_dir / "results").is_dir())
        self.assertEqual(list(self.root.glob("work/autorlbench-run-*")), [])

    def test_issue_only_task_reruns_on_the_local_backend(self) -> None:
        # The submitter command defaults to solution, but a task package may only ship
        # an issue evaluation. Its level-neutral run_spec must still execute, and local
        # prepare must use the staged environment path rather than Docker's mount path.
        from autorlbench.core.types import EvaluationConfig
        from autorlbench.evaluation.evaluator import TaskEvaluator
        from tests.test_harbor_evaluator import create_task

        task_root = create_task(self.root, "issue_only_fixture")
        shutil.rmtree(task_root / "evaluations" / "solution")
        (task_root / "workspace" / "train.sh").write_text(
            "#!/usr/bin/env bash\necho packaged-rerun-ran\n",
            encoding="utf-8",
        )
        write_spec(
            task_root,
            {
                **VALID,
                "command": "bash train.sh",
                "env": {},
                "timeout_s": 60,
            },
        )
        work_dir = self.root / "work"

        result = TaskEvaluator("local").rerun(
            task_root,
            EvaluationConfig(
                backend="local",
                work_dir=work_dir,
                models_dir=self.root / "models",
                datasets_dir=self.root / "datasets",
            ),
        )

        run_dir = next((work_dir / "issue_only_fixture").glob("*"))
        self.assertTrue(result.passed)
        self.assertEqual(result.evaluation_level, "issue")
        self.assertEqual(result.metrics["exit_code"], 0)
        self.assertIn(
            "packaged-rerun-ran",
            (run_dir / "rerun.log").read_text(encoding="utf-8"),
        )


class ShippedRunSpecTests(unittest.TestCase):
    """Every run_spec.json in tasks/ has to load, and declare what the protocol needs."""

    def test_every_shipped_spec_loads(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        # Both locations count: the task root is the current layout, and the legacy
        # evaluations/solution/verifier/resources/ files still load through the fallback.
        task_roots = sorted(repo_root.glob("tasks/*/*"))
        self.assertTrue(task_roots, "no tasks found under tasks/")
        seen = 0
        for task_root in task_roots:
            if not (task_root / "task.toml").is_file():
                continue
            with self.subTest(str(task_root.relative_to(repo_root))):
                spec = load_run_spec(task_root, "shipped")
                if spec is None:
                    continue
                seen += 1
                self.assertTrue(spec.command.strip())
                # log_roots may legitimately be empty for a protocol-check task that produces
                # no experiment logs to clear (for pure protocol-check tasks).
                self.assertIsInstance(spec.log_roots, list)
                # A rerun that trains needs a real budget; falling back to the generic
                # EvaluationConfig timeout would cut a multi-hour run off at the knees.
                self.assertIsNotNone(spec.timeout_s)
        self.assertTrue(
            seen, "no run_spec.json found under tasks/ (root or legacy location)"
        )


if __name__ == "__main__":
    unittest.main()
