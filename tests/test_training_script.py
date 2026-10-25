"""metadata.training.script: the task's training entry point, declared by the task rather than guessed by glob.

docker/debug-env.sh's `run` used to find training scripts with the glob
`workspace/examples/*/{*_rl,run_*}.sh`. That pattern describes slime's layout: RLVE's entry point is
`scripts/training/<model>/rlve.sh`, which never matched, so `run` could not start for those tasks at
all. The entry point now lives on the task card, and this checks that the path itself holds up.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autorlbench.tasks import load_task_package
from autorlbench.tasks.validator import validate_metadata


METADATA = """title = "Training task"
license = "MIT"
created_date = "2026-09-23"
domain = "rl"

[submitter]
name = "test"

[model]
name = "Qwen2.5-0.5B-Instruct"

[dataset]
name = "test"

[environment]
conda = "verl"

[outcome]
failure_type = "test"
failure_reason = "test"

[training]
{training}"""


def create_task(root: Path, training: str, *, launch: str | None = "train.sh") -> Path:
    """Lay out a minimal task; the caller supplies the training section."""
    task = root / "training-task"
    for directory in ("workspace", "oracle", "environment", "verifier"):
        (task / directory).mkdir(parents=True, exist_ok=True)
    (task / "instruction.md").write_text("Fix it.\n", encoding="utf-8")
    (task / "verifier" / "test.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
    )
    if launch is not None:
        path = task / "workspace" / launch
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (task / "task.toml").write_text(
        METADATA.format(training=training), encoding="utf-8"
    )
    return task


class TrainingScriptPathTests(unittest.TestCase):
    """The path half needs TaskPaths to decide, so go through the full load_task_package."""

    def _issues(self, training: str, **kwargs) -> list:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        package = load_task_package(create_task(Path(temp.name), training, **kwargs))
        return [
            (issue.severity, issue.path)
            for issue in package.report.issues
            if issue.path.startswith("metadata.training")
        ]

    def test_declared_script_that_exists_is_clean(self) -> None:
        self._assert_clean(
            'framework = "verl"\nalgo = "grpo"\nscript = "examples/run.sh"\n',
            launch="examples/run.sh",
        )

    def test_script_outside_examples_is_just_as_valid(self) -> None:
        # This is the kind glob could not find: RLVE puts its entry point under scripts/training/.
        self._assert_clean(
            'framework = "slime"\nalgo = "grpo"\n'
            'script = "scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh"\n',
            launch="scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh",
        )

    def test_args_ride_along_with_the_script(self) -> None:
        # rlve.sh exits immediately when `$# -lt 3`: without arguments, that path cannot run.
        self._assert_clean(
            'framework = "slime"\nalgo = "grpo"\nscript = "train.sh"\n'
            'script_args = ["autorlbench", "solution_rerun", "_"]\n'
        )

    def test_missing_file_is_an_error_not_a_surprise_at_run_time(self) -> None:
        # A wrong path must fail at validation; waiting until `run` is often hours later.
        self.assertEqual(
            self._issues('framework = "verl"\nalgo = "grpo"\nscript = "nope.sh"\n'),
            [("error", "metadata.training.script")],
        )

    def test_script_must_stay_inside_the_workspace(self) -> None:
        for script in ("/opt/train.sh", "../train.sh"):
            with self.subTest(script=script):
                self.assertEqual(
                    self._issues(f'framework = "verl"\nalgo = "grpo"\nscript = "{script}"\n'),
                    [("error", "metadata.training.script")],
                )

    def test_missing_script_is_a_warning_when_the_task_trains(self) -> None:
        issues = self._issues('framework = "verl"\nalgo = "grpo"\n')
        self.assertEqual(issues, [("warning", "metadata.training.script")])

    def test_a_task_without_training_needs_no_script(self) -> None:
        # Pure code-fix tasks (framework: none) have no training script.
        self._assert_clean('framework = "none"\nalgo = "none"\n', launch=None)

    def _assert_clean(self, training: str, **kwargs) -> None:
        self.assertEqual(self._issues(training, **kwargs), [])


class TrainingScriptArgsTests(unittest.TestCase):
    """Shape of script_args; no filesystem needed."""

    def _issues(self, training: dict) -> list:
        report = validate_metadata({"training": training})
        return [
            (issue.severity, issue.path)
            for issue in report.issues
            if issue.path.startswith("metadata.training.script")
        ]

    def test_a_string_is_rejected_rather_than_split(self) -> None:
        # Whether "a b c" is one argument or three depends on who splits it. Accepting only a list means readers need not guess.
        self.assertEqual(
            self._issues({"framework": "verl", "algo": "grpo", "script": "t.sh",
                          "script_args": "autorlbench solution_rerun _"}),
            [("error", "metadata.training.script_args")],
        )

    def test_unquoted_number_is_rejected_rather_than_coerced(self) -> None:
        # In TOML `0` is an int; silently calling str() would hide a mistake on the task card.
        self.assertEqual(
            self._issues({"framework": "verl", "algo": "grpo", "script": "t.sh",
                          "script_args": ["autorlbench", 0]}),
            [("error", "metadata.training.script_args[1]")],
        )

    def test_args_without_a_script_have_nothing_to_apply_to(self) -> None:
        self.assertEqual(
            self._issues({"framework": "none", "algo": "none", "script_args": ["x"]}),
            [("error", "metadata.training.script_args")],
        )


if __name__ == "__main__":
    unittest.main()
